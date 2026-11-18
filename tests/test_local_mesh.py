# generated-by: gemini-ultra
"""TDD test suite for oddly-local-mesh (GGUF math, real 127.0.0.1 TCP probes, roofline planner, Wi-Fi detector, node-drop replanner, and llama.cpp RPC commands)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_mesh.cli import main as cli_main
from local_mesh.commands import generate_llama_cpp_commands
from local_mesh.models import (
    estimate_model_memory,
    get_model_spec,
)
from local_mesh.planner import (
    NodeSpec,
    load_cluster_preset,
    plan_cluster,
    replan_after_drop,
)
from local_mesh.probe import (
    BeaconListener,
    ProbeServer,
    probe_tcp_peer,
    send_udp_beacon,
)


def _three_laptop_nodes() -> list[NodeSpec]:
    return [
        NodeSpec(
            name="thinkpad-t480",
            host="thinkpad-t480.local",
            rpc_port=50052,
            role="coordinator",
            ram_gib=16.0,
            mem_bw_gbps=32.0,
            compute_tflops=1.2,
            link_type="ethernet-1g",
            rtt_ms=0.0,
            bandwidth_mbps=10000.0,
        ),
        NodeSpec(
            name="latitude-7490",
            host="latitude-7490.local",
            rpc_port=50052,
            role="worker",
            ram_gib=16.0,
            mem_bw_gbps=32.0,
            compute_tflops=1.2,
            link_type="ethernet-1g",
            rtt_ms=0.38,
            bandwidth_mbps=940.0,
        ),
        NodeSpec(
            name="macbook-air-m1",
            host="macbook-air-m1.local",
            rpc_port=50052,
            role="worker",
            ram_gib=8.0,
            mem_bw_gbps=68.0,
            compute_tflops=5.2,
            link_type="ethernet-1g",
            rtt_ms=0.35,
            bandwidth_mbps=940.0,
        ),
    ]


# --- 1. GGUF Model & KV-Cache Byte Formula Tests ---


def test_model_catalog_covers_4b_14b_30b_70b() -> None:
    for alias in ("4b", "14b", "30b", "70b"):
        spec = get_model_spec(alias)
        assert spec.n_layers >= 32
        assert spec.params_b > 0


def test_quantization_footprint_monotonic_q4_q5_q8_f16() -> None:
    m_q4 = estimate_model_memory("30b", quant="Q4_K_M", n_ctx=4096)
    m_q5 = estimate_model_memory("30b", quant="Q5_K_M", n_ctx=4096)
    m_q8 = estimate_model_memory("30b", quant="Q8_0", n_ctx=4096)
    m_f16 = estimate_model_memory("30b", quant="F16", n_ctx=4096)
    assert m_q4.total_weight_gib < m_q5.total_weight_gib < m_q8.total_weight_gib < m_f16.total_weight_gib


def test_kv_cache_scales_linearly_with_context_and_kv_quantization() -> None:
    kv_4k_f16 = estimate_model_memory("30b", quant="Q4_K_M", n_ctx=4096, kv_type="f16")
    kv_16k_f16 = estimate_model_memory("30b", quant="Q4_K_M", n_ctx=16384, kv_type="f16")
    kv_16k_q8 = estimate_model_memory("30b", quant="Q4_K_M", n_ctx=16384, kv_type="q8_0")

    assert abs(kv_16k_f16.per_layer_kv_gib - 4.0 * kv_4k_f16.per_layer_kv_gib) < 1e-6
    assert kv_16k_q8.per_layer_kv_gib < kv_16k_f16.per_layer_kv_gib


def test_30b_q4_exceeds_single_16gb_laptop_with_os_headroom() -> None:
    fp = estimate_model_memory("30b", quant="Q4_K_M", n_ctx=4096)
    # 30B/32B Q4_K_M is ~19-20 GiB total, which cannot fit inside 16 GiB - 2.5 GiB OS headroom
    assert fp.total_required_gib > 13.5


# --- 2. Real Multi-Port 127.0.0.1 TCP Socket Micro-Probe & UDP Beacon Tests ---


def test_real_multi_port_loopback_tcp_probe_measures_rtt_and_throughput() -> None:
    with ProbeServer(host="127.0.0.1", port=0) as srv_a, ProbeServer(
        host="127.0.0.1", port=0
    ) as srv_b:
        assert srv_a.bound_port > 0
        assert srv_b.bound_port > 0
        assert srv_a.bound_port != srv_b.bound_port

        m_a = probe_tcp_peer("127.0.0.1", srv_a.bound_port, iters=12, slice_bytes=131072)
        m_b = probe_tcp_peer("127.0.0.1", srv_b.bound_port, iters=12, slice_bytes=131072)

        assert m_a.reachable is True
        assert m_b.reachable is True
        assert 0.001 < m_a.rtt_median_ms < 50.0
        assert m_a.rtt_p95_ms >= m_a.rtt_median_ms
        assert m_a.bandwidth_mbps > 50.0
        assert m_b.bandwidth_mbs > 5.0


def test_probe_unreachable_port_returns_clean_failure_without_exception() -> None:
    # Bind and close immediately to obtain a closed port on 127.0.0.1
    with ProbeServer(host="127.0.0.1", port=0) as temp_srv:
        closed_port = temp_srv.bound_port
    metrics = probe_tcp_peer("127.0.0.1", closed_port, iters=3, timeout_sec=0.3)
    assert metrics.reachable is False
    assert metrics.error is not None


def test_udp_loopback_beacon_announces_and_discovers_peer() -> None:
    with BeaconListener(host="127.0.0.1", port=0) as listener:
        payload = {
            "name": "mini-pc-a",
            "host": "127.0.0.1",
            "rpc_port": 50052,
            "ram_gib": 16.0,
        }
        send_udp_beacon(payload, host="127.0.0.1", port=listener.bound_port)
        discovered = listener.collect(timeout_sec=0.5, max_peers=1)
        assert len(discovered) == 1
        assert discovered[0]["name"] == "mini-pc-a"
        assert discovered[0]["rpc_port"] == 50052


# --- 3. Layer Planner, OS Headroom, Slow Wi-Fi Exclusion & OOM Rejection Tests ---


def test_three_laptop_30b_q4_split_fits_16gb_16gb_8gb_with_os_headroom() -> None:
    nodes = _three_laptop_nodes()
    plan = plan_cluster(nodes, model="30b", quant="Q4_K_M", n_ctx=4096, os_headroom_gib=2.5)

    assert plan.fits is True
    assert len(plan.allocations) == 3
    total_layers = sum(a.assigned_layers for a in plan.allocations)
    assert total_layers == get_model_spec("30b").n_layers

    # Every single laptop uses strictly <= usable_ram_gib and preserves >= 2.5 GiB OS headroom
    for alloc, orig in zip(plan.allocations, nodes):
        assert alloc.assigned_layers > 0
        assert alloc.used_ram_gib <= alloc.usable_ram_gib + 1e-6
        assert (orig.ram_gib - alloc.used_ram_gib) >= 2.5 - 1e-6

    assert plan.decode_tok_s > 0.5
    assert plan.decode_bounds_tok_s[0] < plan.decode_tok_s < plan.decode_bounds_tok_s[1]
    assert plan.ttft_sec > 0.1


def test_oom_rejection_when_model_exceeds_cluster_ram() -> None:
    nodes = _three_laptop_nodes()  # 16 + 16 + 8 = 40 GiB raw, ~30.6 GiB usable
    # 70B Q4_K_M requires ~42+ GiB -> must cleanly reject as OOM!
    plan = plan_cluster(nodes, model="70b", quant="Q4_K_M", n_ctx=4096, os_headroom_gib=2.5)
    assert plan.fits is False
    assert plan.deficit_gib > 5.0
    assert plan.fallback_suggestion is not None
    assert "30b" in plan.fallback_suggestion.lower() or "14b" in plan.fallback_suggestion.lower()


def test_slow_wifi_node_excluded_when_adding_it_hurts_decode_tok_s() -> None:
    # Two wired 16 GB laptops already hold ~24.5 GiB usable (> ~19.5 GiB needed for 30B Q4_K_M).
    # Adding a slow Wi-Fi 8 GB laptop (RTT 38 ms, 35 Mbps, 12 GB/s DDR3) hurts tok/s!
    nodes = [
        NodeSpec(
            name="thinkpad-wired-a",
            host="thinkpad-a.local",
            rpc_port=50052,
            role="coordinator",
            ram_gib=16.0,
            mem_bw_gbps=38.0,
            compute_tflops=1.8,
            link_type="ethernet-1g",
            rtt_ms=0.0,
            bandwidth_mbps=10000.0,
        ),
        NodeSpec(
            name="thinkpad-wired-b",
            host="thinkpad-b.local",
            rpc_port=50052,
            role="worker",
            ram_gib=16.0,
            mem_bw_gbps=38.0,
            compute_tflops=1.8,
            link_type="ethernet-1g",
            rtt_ms=0.35,
            bandwidth_mbps=940.0,
        ),
        NodeSpec(
            name="old-wifi-notebook",
            host="old-wifi-notebook.local",
            rpc_port=50052,
            role="worker",
            ram_gib=8.0,
            mem_bw_gbps=12.0,
            compute_tflops=0.5,
            link_type="wifi-2.4ghz",
            rtt_ms=38.0,
            bandwidth_mbps=35.0,
        ),
    ]
    plan = plan_cluster(nodes, model="30b", quant="Q4_K_M", n_ctx=4096, os_headroom_gib=2.5)
    assert plan.fits is True
    assert "old-wifi-notebook" in plan.excluded_nodes
    active_names = [a.name for a in plan.allocations if a.assigned_layers > 0]
    assert active_names == ["thinkpad-wired-a", "thinkpad-wired-b"]
    assert any("old-wifi-notebook" in w for w in plan.warnings)


def test_slow_wifi_node_kept_with_warning_when_strictly_needed_to_avoid_oom() -> None:
    # 12 GB + 12 GB wired = ~17.5 GiB usable (< ~19.5 GiB needed for 30B Q4_K_M), so 8 GB Wi-Fi laptop IS needed to fit!
    nodes = [
        NodeSpec(
            name="laptop-a",
            host="laptop-a.local",
            role="coordinator",
            ram_gib=12.0,
            mem_bw_gbps=32.0,
            link_type="ethernet-1g",
            rtt_ms=0.0,
            bandwidth_mbps=10000.0,
        ),
        NodeSpec(
            name="laptop-b",
            host="laptop-b.local",
            role="worker",
            ram_gib=12.0,
            mem_bw_gbps=32.0,
            link_type="ethernet-1g",
            rtt_ms=0.4,
            bandwidth_mbps=940.0,
        ),
        NodeSpec(
            name="wifi-helper",
            host="wifi-helper.local",
            role="worker",
            ram_gib=8.0,
            mem_bw_gbps=20.0,
            link_type="wifi-5ghz",
            rtt_ms=14.0,
            bandwidth_mbps=180.0,
        ),
    ]
    plan = plan_cluster(nodes, model="30b", quant="Q4_K_M", n_ctx=4096, os_headroom_gib=2.5)
    assert plan.fits is True
    assert "wifi-helper" not in plan.excluded_nodes
    assert any("wi-fi" in w.lower() or "wifi" in w.lower() for w in plan.warnings)


def test_single_fast_coordinator_preferred_when_model_fits_locally() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    nodes, headroom = load_cluster_preset(repo_root / "presets" / "mac_plus_two_minis_70b.json")
    # 14B Q4_K_M (~9.6 GiB) easily fits on 32 GB Mac Studio alone (200 GB/s) without RPC hops!
    plan = plan_cluster(nodes, model="14b", quant="Q4_K_M", n_ctx=4096, os_headroom_gib=headroom)
    assert plan.fits is True
    active = [a for a in plan.allocations if a.assigned_layers > 0]
    assert len(active) == 1
    assert active[0].name == "mac-studio"


def test_mac_plus_two_minis_preset_fits_70b_q4_across_all_three_nodes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    nodes, headroom = load_cluster_preset(repo_root / "presets" / "mac_plus_two_minis_70b.json")
    plan = plan_cluster(nodes, model="70b", quant="Q4_K_M", n_ctx=4096, os_headroom_gib=headroom)
    assert plan.fits is True
    active = [a for a in plan.allocations if a.assigned_layers > 0]
    assert len(active) == 3
    assert sum(a.assigned_layers for a in active) == 80


# --- 4. Fault-Tolerant Node-Drop Replanner Tests ---


def test_replan_after_drop_rebalances_onto_surviving_nodes_when_ram_suffices() -> None:
    nodes = _three_laptop_nodes()
    outcome = replan_after_drop(
        nodes,
        dropped_names=["macbook-air-m1"],
        model="30b",
        quant="Q4_K_M",
        n_ctx=4096,
        os_headroom_gib=2.5,
    )
    assert outcome["status"] == "REBALANCED"
    new_plan = outcome["plan"]
    assert new_plan.fits is True
    active_names = [a.name for a in new_plan.allocations if a.assigned_layers > 0]
    assert active_names == ["thinkpad-t480", "latitude-7490"]
    assert sum(a.assigned_layers for a in new_plan.allocations) == 64


def test_replan_after_drop_recommends_smaller_quant_when_q8_no_longer_fits() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    nodes, headroom = load_cluster_preset(repo_root / "presets" / "mac_plus_two_minis_70b.json")
    # 30B Q8_0 (~34.5 GiB) fits on Mac Studio (32G) + 2 Minis (16G+16G),
    # but if BOTH minis drop, Mac Studio alone (~28.7 GiB usable) cannot fit 30B Q8_0 -> falls back to Q4_K_M!
    outcome = replan_after_drop(
        nodes,
        dropped_names=["mini-pc-a", "mini-pc-b"],
        model="30b",
        quant="Q8_0",
        n_ctx=4096,
        os_headroom_gib=headroom,
    )
    assert outcome["status"] == "DOWNGRADE_REQUIRED"
    fallback_plan = outcome["fallback_plan"]
    assert fallback_plan is not None
    assert fallback_plan.fits is True
    assert fallback_plan.model_key == "30b"
    assert fallback_plan.quant in ("Q5_K_M", "Q4_K_M")


def test_replan_after_coordinator_drop_promotes_surviving_worker() -> None:
    nodes = _three_laptop_nodes()
    outcome = replan_after_drop(
        nodes,
        dropped_names=["thinkpad-t480"],
        model="14b",
        quant="Q4_K_M",
        n_ctx=4096,
    )
    assert outcome["status"] == "REBALANCED"
    assert outcome["promoted_coordinator"] == "latitude-7490"


# --- 5. llama.cpp RPC Command Generator & OPSEC Preset Tests ---


def test_generated_llama_server_rpc_and_tensor_split_flags_match_active_workers() -> None:
    nodes = _three_laptop_nodes()
    plan = plan_cluster(nodes, model="30b", quant="Q4_K_M", n_ctx=4096, os_headroom_gib=2.5)
    cmds = generate_llama_cpp_commands(plan, model_path="models/qwen2.5-32b-q4_k_m.gguf")

    assert len(cmds["worker_commands"]) == 2
    assert "rpc-server --host latitude-7490.local -p 50052" in cmds["worker_commands"][0]["command"]
    assert "rpc-server --host macbook-air-m1.local -p 50052" in cmds["worker_commands"][1]["command"]

    server_cmd = cmds["coordinator_server_cmd"]
    assert "--rpc latitude-7490.local:50052,macbook-air-m1.local:50052" in server_cmd
    assert "--split-mode layer" in server_cmd
    assert "--tensor-split " in server_cmd
    assert "--host 127.0.0.1" in server_cmd
    assert any("unauthenticated" in note.lower() for note in cmds["security_notes"])


def test_generated_commands_omit_rpc_flag_when_single_node_selected() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    nodes, headroom = load_cluster_preset(repo_root / "presets" / "mac_plus_two_minis_70b.json")
    plan = plan_cluster(nodes, model="14b", quant="Q4_K_M", n_ctx=4096, os_headroom_gib=headroom)
    cmds = generate_llama_cpp_commands(plan)
    assert cmds["worker_commands"] == []
    assert "--rpc" not in cmds["coordinator_server_cmd"]


def test_presets_contain_zero_private_lan_ip_addresses() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for preset_path in (repo_root / "presets").glob("*.json"):
        raw = preset_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert parsed["_generated_by"] == "gemini-ultra"
        for node in parsed["nodes"]:
            host = str(node["host"])
            assert host == "127.0.0.1" or host.endswith(".local")


# --- 6. CLI End-to-End Tests ---


def test_cli_plan_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    preset = repo_root / "presets" / "three_old_laptops_30b.json"
    rc = cli_main(["plan", "--preset", str(preset), "--model", "30b", "--quant", "Q4_K_M", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["fits"] is True
    assert len(out["commands"]["worker_commands"]) == 2


def test_cli_plan_returns_nonzero_on_oom() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    preset = repo_root / "presets" / "three_old_laptops_30b.json"
    rc = cli_main(["plan", "--preset", str(preset), "--model", "70b", "--quant", "Q8_0"])
    assert rc == 2


def test_cli_simulate_drop_human_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    preset = repo_root / "presets" / "three_old_laptops_30b.json"
    rc = cli_main(
        [
            "simulate-drop",
            "--preset",
            str(preset),
            "--drop",
            "macbook-air-m1",
            "--model",
            "30b",
            "--quant",
            "Q4_K_M",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "REBALANCED"


def test_cli_probe_against_live_loopback_servers(capsys: pytest.CaptureFixture[str]) -> None:
    with ProbeServer("127.0.0.1", 0) as s1, ProbeServer("127.0.0.1", 0) as s2:
        targets = f"127.0.0.1:{s1.bound_port},127.0.0.1:{s2.bound_port}"
        rc = cli_main(["probe", "--targets", targets, "--iters", "6", "--slice-kb", "64", "--json"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert len(out["results"]) == 2
        assert all(r["reachable"] for r in out["results"])


def test_cli_bench_runs_end_to_end_loopback_cluster(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(["bench", "--nodes", "3", "--iters", "8", "--slice-kb", "64", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["_generated_by"] == "gemini-ultra"
    assert len(out["measured_links"]) == 2
    assert out["sample_plan_30b_q4"]["fits"] is True
