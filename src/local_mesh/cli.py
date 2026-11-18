# generated-by: gemini-ultra
"""Command-line interface for `local-mesh` (`plan`, `probe`, `simulate-drop`, `bench`)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from contextlib import ExitStack
import json
import sys
from typing import Any

from local_mesh.commands import generate_llama_cpp_commands
from local_mesh.planner import (
    ClusterPlan,
    NodeSpec,
    load_cluster_preset,
    plan_cluster,
    replan_after_drop,
)
from local_mesh.probe import ProbeServer, probe_tcp_peer


def _format_plan_text(plan: ClusterPlan, cmds: dict[str, Any]) -> str:
    lines = [
        f"=== local-mesh Roofline Plan: {plan.display_name} ({plan.quant}, n_ctx={plan.n_ctx}, kv={plan.kv_type}) ===",
        f"Model Footprint (weights + KV + scratch): {plan.total_required_gib:.2f} GiB | Cluster Usable RAM: {plan.cluster_usable_gib:.2f} GiB",
    ]
    if not plan.fits:
        lines.append(f"[OOM REJECTED] Deficit: {plan.deficit_gib:.2f} GiB")
        for w in plan.warnings:
            lines.append(f"  ! {w}")
        if plan.fallback_suggestion:
            lines.append(f"  -> Suggestion: {plan.fallback_suggestion}")
        return "\n".join(lines)

    lines.append(
        f"Honest Speed Estimate: Decode = {plan.decode_tok_s:.2f} tok/s "
        f"(bounds {plan.decode_bounds_tok_s[0]:.2f}–{plan.decode_bounds_tok_s[1]:.2f} tok/s) | "
        f"TTFT (512 tok) = {plan.ttft_sec:.2f}s "
        f"(bounds {plan.ttft_bounds_sec[0]:.2f}–{plan.ttft_bounds_sec[1]:.2f}s)"
    )
    lines.append("")
    lines.append("Node Allocations (OS headroom preserved):")
    for a in plan.allocations:
        if a.assigned_layers == 0 and a.name in plan.excluded_nodes:
            lines.append(
                f"  - {a.name:18s} [{a.role:11s}] EXCLUDED (slow link / unnecessary hop)"
            )
        else:
            l_str = (
                f"layers {a.layer_range[0]:02d}..{a.layer_range[1]:02d} ({a.assigned_layers:2d})"
                if a.layer_range is not None
                else "0 layers"
            )
            lines.append(
                f"  - {a.name:18s} [{a.role:11s}] {l_str} | "
                f"RAM {a.used_ram_gib:5.2f}/{a.usable_ram_gib:5.2f} GiB usable "
                f"(free OS headroom: {a.headroom_gib:5.2f} GiB)"
            )

    if plan.warnings:
        lines.append("")
        lines.append("Planner Warnings:")
        for w in plan.warnings:
            lines.append(f"  [!] {w}")

    lines.append("")
    lines.append("Ready-to-Run llama.cpp Commands:")
    for wc in cmds["worker_commands"]:
        lines.append(f"  # On worker {wc['node']}:")
        lines.append(f"  {wc['command']}")
    lines.append("  # On coordinator:")
    lines.append(f"  {cmds['coordinator_server_cmd']}")
    lines.append("")
    lines.append(f"  [{cmds['security_notes'][0]}]")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-mesh",
        description="Honest LAN llama.cpp RPC layer planner, TCP micro-prober, and roofline speed simulator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="Compute RAM-safe layer split, roofline tok/s, and llama.cpp flags")
    p_plan.add_argument("--preset", required=True, help="Path to cluster preset JSON")
    p_plan.add_argument("--model", default="30b", help="Model key: 4b, 14b, 30b, 70b")
    p_plan.add_argument("--quant", default="Q4_K_M", help="Quantization: Q4_K_M, Q5_K_M, Q8_0, F16")
    p_plan.add_argument("--ctx", type=int, default=4096, help="Context window tokens (default: 4096)")
    p_plan.add_argument("--kv-type", default="f16", help="KV cache precision: f16, q8_0, q4_0")
    p_plan.add_argument("--prompt-tokens", type=int, default=512, help="Prompt length for TTFT estimate")
    p_plan.add_argument("--model-path", help="Optional GGUF path for generated commands")
    p_plan.add_argument("--json", action="store_true", help="Emit JSON output")

    p_probe = sub.add_parser("probe", help="Run real TCP_NODELAY RTT and tensor-slice throughput probes")
    p_probe.add_argument(
        "--targets",
        required=True,
        help="Comma-separated host:port list (e.g. 127.0.0.1:50052,127.0.0.1:50053)",
    )
    p_probe.add_argument("--iters", type=int, default=20, help="Ping-pong RTT iterations")
    p_probe.add_argument("--slice-kb", type=int, default=256, help="Activation slice payload KiB")
    p_probe.add_argument("--json", action="store_true", help="Emit JSON output")

    p_drop = sub.add_parser(
        "simulate-drop",
        help="Simulate one or more nodes dropping mid-session and compute live rebalance/fallback",
    )
    p_drop.add_argument("--preset", required=True, help="Path to cluster preset JSON")
    p_drop.add_argument("--drop", required=True, help="Comma-separated dropped node names")
    p_drop.add_argument("--model", default="30b", help="Model key: 4b, 14b, 30b, 70b")
    p_drop.add_argument("--quant", default="Q4_K_M", help="Quantization: Q4_K_M, Q5_K_M, Q8_0, F16")
    p_drop.add_argument("--ctx", type=int, default=4096, help="Context window tokens")
    p_drop.add_argument("--json", action="store_true", help="Emit JSON output")

    p_bench = sub.add_parser(
        "bench",
        help="Spin up real local 127.0.0.1 TCP probe workers, measure socket speed, and run a 30B plan",
    )
    p_bench.add_argument("--nodes", type=int, default=3, help="Total nodes including coordinator (>=2)")
    p_bench.add_argument("--iters", type=int, default=25, help="RTT probe iterations")
    p_bench.add_argument("--slice-kb", type=int, default=256, help="Tensor slice KiB")
    p_bench.add_argument("--json", action="store_true", help="Emit JSON output")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "plan":
        nodes, headroom = load_cluster_preset(args.preset)
        plan = plan_cluster(
            nodes,
            model=args.model,
            quant=args.quant,
            n_ctx=args.ctx,
            kv_type=args.kv_type,
            os_headroom_gib=headroom,
            prompt_tokens=args.prompt_tokens,
        )
        cmds = generate_llama_cpp_commands(plan, model_path=args.model_path)
        if args.json:
            payload = asdict(plan)
            payload["_generated_by"] = "gemini-ultra"
            payload["commands"] = cmds
            print(json.dumps(payload, indent=2))
        else:
            print(_format_plan_text(plan, cmds))
        return 0 if plan.fits else 2

    if args.command == "probe":
        results = []
        slice_bytes = max(1024, args.slice_kb * 1024)
        for item in args.targets.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                print(f"[ERROR] Invalid target '{item}', expected host:port", file=sys.stderr)
                return 2
            host, port_s = item.rsplit(":", 1)
            metrics = probe_tcp_peer(host, int(port_s), iters=args.iters, slice_bytes=slice_bytes)
            results.append(asdict(metrics))

        if args.json:
            print(json.dumps({"_generated_by": "gemini-ultra", "results": results}, indent=2))
        else:
            for r in results:
                if r["reachable"]:
                    print(
                        f"[OK] {r['target']} -> RTT median={r['rtt_median_ms']:.3f} ms "
                        f"(p95={r['rtt_p95_ms']:.3f} ms), slice throughput={r['bandwidth_mbs']:.1f} MB/s "
                        f"({r['bandwidth_mbps']:.1f} Mbps)"
                    )
                else:
                    print(f"[UNREACHABLE] {r['target']} -> {r['error']}")
        return 0 if all(r["reachable"] for r in results) else 2

    if args.command == "simulate-drop":
        nodes, headroom = load_cluster_preset(args.preset)
        dropped = [d.strip() for d in args.drop.split(",") if d.strip()]
        outcome = replan_after_drop(
            nodes,
            dropped_names=dropped,
            model=args.model,
            quant=args.quant,
            n_ctx=args.ctx,
            os_headroom_gib=headroom,
        )
        if args.json:
            serializable = {
                "_generated_by": "gemini-ultra",
                "status": outcome["status"],
                "dropped": outcome["dropped"],
                "promoted_coordinator": outcome["promoted_coordinator"],
                "message": outcome["message"],
                "plan": asdict(outcome["plan"]) if outcome["plan"] else None,
                "fallback_plan": (
                    asdict(outcome["fallback_plan"]) if outcome["fallback_plan"] else None
                ),
            }
            print(json.dumps(serializable, indent=2))
        else:
            print(f"[{outcome['status']}] {outcome['message']}")
            active_plan = (
                outcome["plan"]
                if outcome["status"] == "REBALANCED"
                else outcome["fallback_plan"]
            )
            if active_plan is not None and active_plan.fits:
                cmds = generate_llama_cpp_commands(active_plan)
                print("")
                print(_format_plan_text(active_plan, cmds))
        return 0

    if args.command == "bench":
        n_total = max(2, args.nodes)
        slice_bytes = max(1024, args.slice_kb * 1024)
        with ExitStack() as stack:
            servers = [
                stack.enter_context(ProbeServer(host="127.0.0.1", port=0))
                for _ in range(n_total - 1)
            ]
            measured_links = [
                probe_tcp_peer("127.0.0.1", s.bound_port, iters=args.iters, slice_bytes=slice_bytes)
                for s in servers
            ]

        bench_nodes: list[NodeSpec] = [
            NodeSpec(
                name="node-0-coord",
                host="127.0.0.1",
                rpc_port=50051,
                role="coordinator",
                ram_gib=16.0,
                mem_bw_gbps=32.0,
                compute_tflops=1.2,
                link_type="loopback",
                rtt_ms=0.0,
                bandwidth_mbps=20000.0,
            )
        ]
        for idx, m in enumerate(measured_links, start=1):
            ram = 8.0 if idx == n_total - 1 and n_total >= 3 else 16.0
            bw_ram = 68.0 if ram == 8.0 else 32.0
            bench_nodes.append(
                NodeSpec(
                    name=f"node-{idx}-worker",
                    host="127.0.0.1",
                    rpc_port=int(m.target.rsplit(":", 1)[1]),
                    role="worker",
                    ram_gib=ram,
                    mem_bw_gbps=bw_ram,
                    compute_tflops=5.2 if ram == 8.0 else 1.2,
                    link_type="loopback-tcp",
                    rtt_ms=m.rtt_median_ms,
                    bandwidth_mbps=m.bandwidth_mbps,
                )
            )

        plan = plan_cluster(bench_nodes, model="30b", quant="Q4_K_M", n_ctx=4096)
        cmds = generate_llama_cpp_commands(plan)
        if args.json:
            print(
                json.dumps(
                    {
                        "_generated_by": "gemini-ultra",
                        "measured_links": [asdict(m) for m in measured_links],
                        "sample_plan_30b_q4": asdict(plan),
                        "commands": cmds,
                    },
                    indent=2,
                )
            )
        else:
            print("=== Measured Local Loopback TCP Socket Probes (127.0.0.1) ===")
            for m in measured_links:
                print(
                    f"  - {m.target}: RTT median={m.rtt_median_ms:.3f} ms "
                    f"(p95={m.rtt_p95_ms:.3f} ms), throughput={m.bandwidth_mbs:.1f} MB/s "
                    f"({m.bandwidth_mbps:.1f} Mbps)"
                )
            print("")
            print(_format_plan_text(plan, cmds))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
