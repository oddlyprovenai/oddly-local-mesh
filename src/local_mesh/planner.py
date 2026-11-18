# generated-by: gemini-ultra
"""RAM/KV-safe layer allocator, honest roofline TTFT/decode simulator, slow-Wi-Fi detector, and node-drop replanner."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import json
from pathlib import Path
from typing import Any

from local_mesh.models import (
    GIB,
    MemoryFootprint,
    estimate_model_memory,
    get_model_spec,
)


@dataclass(frozen=True)
class NodeSpec:
    """Hardware and network link profile of a cluster participant."""

    name: str
    host: str
    rpc_port: int = 50052
    role: str = "worker"  # "coordinator" or "worker"
    ram_gib: float = 16.0
    mem_bw_gbps: float = 32.0
    compute_tflops: float = 1.5
    link_type: str = "ethernet-1g"
    rtt_ms: float = 0.4
    bandwidth_mbps: float = 940.0


@dataclass(frozen=True)
class NodeAllocation:
    """Layer assignment and memory budget for one node in a planned cluster."""

    name: str
    host: str
    rpc_port: int
    role: str
    assigned_layers: int
    layer_range: tuple[int, int] | None
    used_ram_gib: float
    usable_ram_gib: float
    headroom_gib: float
    tensor_split_weight: float


@dataclass(frozen=True)
class ClusterPlan:
    """Complete roofline & RAM allocation result for a cluster + model configuration."""

    fits: bool
    model_key: str
    display_name: str
    quant: str
    n_ctx: int
    kv_type: str
    total_required_gib: float
    cluster_usable_gib: float
    deficit_gib: float
    allocations: list[NodeAllocation]
    excluded_nodes: list[str]
    decode_tok_s: float
    decode_bounds_tok_s: tuple[float, float]
    ttft_sec: float
    ttft_bounds_sec: tuple[float, float]
    warnings: list[str] = field(default_factory=list)
    fallback_suggestion: str | None = None


def load_cluster_preset(path: str | Path) -> tuple[list[NodeSpec], float]:
    """Load `NodeSpec` list and `os_headroom_gib` from a JSON preset file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    os_headroom = float(raw.get("os_headroom_gib", 2.5))
    nodes: list[NodeSpec] = []
    for item in raw.get("nodes", []):
        nodes.append(
            NodeSpec(
                name=str(item["name"]),
                host=str(item["host"]),
                rpc_port=int(item.get("rpc_port", 50052)),
                role=str(item.get("role", "worker")),
                ram_gib=float(item["ram_gib"]),
                mem_bw_gbps=float(item["mem_bw_gbps"]),
                compute_tflops=float(item.get("compute_tflops", 1.5)),
                link_type=str(item.get("link_type", "ethernet-1g")),
                rtt_ms=float(item.get("rtt_ms", 0.4)),
                bandwidth_mbps=float(item.get("bandwidth_mbps", 940.0)),
            )
        )
    if not nodes:
        raise ValueError("Preset contains zero nodes")
    return nodes, os_headroom


def _node_layer_capacity(
    node: NodeSpec,
    fp: MemoryFootprint,
    os_headroom_gib: float,
    is_coordinator: bool,
) -> tuple[float, float, int]:
    usable_gib = max(0.0, node.ram_gib - os_headroom_gib)
    fixed_gib = fp.coordinator_fixed_gib if is_coordinator else fp.worker_scratch_gib
    avail_for_layers = max(0.0, usable_gib - fixed_gib)
    max_layers = int(avail_for_layers // fp.per_layer_total_gib)
    return usable_gib, fixed_gib, max_layers


def _allocate_layers_on_subset(
    subset: list[NodeSpec],
    fp: MemoryFootprint,
    os_headroom_gib: float,
) -> list[int] | None:
    """Assign `fp.n_layers` across `subset` (index 0 is coordinator), prioritizing high-bandwidth RAM."""
    caps: list[int] = []
    for idx, node in enumerate(subset):
        _, _, max_l = _node_layer_capacity(node, fp, os_headroom_gib, is_coordinator=(idx == 0))
        caps.append(max_l)

    if sum(caps) < fp.n_layers:
        return None

    # Group indices by memory bandwidth tier (within 5% treated as same speed tier, balanced proportionally)
    remaining = fp.n_layers
    assigned = [0] * len(subset)

    # Sort distinct bandwidth tiers descending
    indexed = list(enumerate(subset))
    indexed.sort(key=lambda pair: pair[1].mem_bw_gbps, reverse=True)

    groups: list[list[int]] = []
    for idx, node in indexed:
        if not groups:
            groups.append([idx])
        else:
            lead_bw = subset[groups[-1][0]].mem_bw_gbps
            if abs(node.mem_bw_gbps - lead_bw) / max(1e-6, lead_bw) <= 0.05:
                groups[-1].append(idx)
            else:
                groups.append([idx])

    for grp in groups:
        if remaining <= 0:
            break
        grp_cap = sum(caps[i] for i in grp)
        if grp_cap <= 0:
            continue
        take = min(remaining, grp_cap)
        if take == grp_cap:
            for i in grp:
                assigned[i] = caps[i]
        else:
            # Distribute `take` proportionally to capacity within equal-bandwidth tier
            allocated_in_grp = 0
            shares: list[tuple[float, int]] = []
            for i in grp:
                exact = take * (caps[i] / float(grp_cap))
                base = min(caps[i], int(exact))
                assigned[i] = base
                allocated_in_grp += base
                shares.append((exact - base, i))
            shares.sort(reverse=True)
            leftover = take - allocated_in_grp
            for _, i in shares:
                if leftover <= 0:
                    break
                if assigned[i] < caps[i]:
                    assigned[i] += 1
                    leftover -= 1
        remaining -= take

    if remaining > 0:
        return None
    return assigned


def _simulate_roofline(
    subset: list[NodeSpec],
    layer_counts: list[int],
    fp: MemoryFootprint,
    prompt_tokens: int = 512,
) -> tuple[float, float]:
    """Compute honest sequential pipeline decode `tok/s` and prefill `TTFT (s)`."""
    per_layer_weight_bytes = fp.per_layer_weight_gib * GIB
    params_per_layer = (fp.params_b * 1e9) / float(fp.n_layers)

    # Autoregressive decode (batch=1): memory-bandwidth bound + per-hop RTT + activation vector transfer
    activation_vec_bytes = float(fp.hidden_dim) * 2.0  # float16 residual stream vector
    decode_time_sec = 0.0

    # Coordinator embedding + LM head read time
    coord = subset[0]
    decode_time_sec += (fp.embedding_and_head_gib * GIB) / (coord.mem_bw_gbps * 1e9 * 0.78)

    active_workers = 0
    for idx, (node, n_l) in enumerate(zip(subset, layer_counts)):
        if n_l <= 0:
            continue
        eff_mem_bw = node.mem_bw_gbps * 1e9 * 0.78  # realistic sustained DRAM/Metal/CUDA utilization
        eff_flops = max(0.1, node.compute_tflops) * 1e12 * 0.55
        mem_t = (n_l * per_layer_weight_bytes) / eff_mem_bw
        comp_t = (n_l * 2.0 * params_per_layer) / eff_flops
        decode_time_sec += max(mem_t, comp_t) + 0.15 * min(mem_t, comp_t)

        if idx > 0:
            active_workers += 1
            bw_bytes_s = max(1e5, node.bandwidth_mbps * 1_000_000.0 / 8.0)
            # Forward hop to worker + return/chain framing overhead
            decode_time_sec += (node.rtt_ms / 1000.0) + (activation_vec_bytes / bw_bytes_s)

    if active_workers > 0:
        # Final return hop from last worker back to coordinator for norm + sampling
        last_worker = [subset[i] for i in range(1, len(subset)) if layer_counts[i] > 0][-1]
        bw_bytes_s = max(1e5, last_worker.bandwidth_mbps * 1_000_000.0 / 8.0)
        decode_time_sec += (last_worker.rtt_ms / 1000.0) + (activation_vec_bytes / bw_bytes_s)

    decode_tok_s = 1.0 / max(1e-6, decode_time_sec)

    # Prefill TTFT (compute-bound matrix-matrix + activation matrix [prompt_tokens, hidden_dim] transfers)
    prefill_act_bytes = float(prompt_tokens) * float(fp.hidden_dim) * 2.0
    ttft_sec = 0.0
    for idx, (node, n_l) in enumerate(zip(subset, layer_counts)):
        if n_l <= 0:
            continue
        eff_flops = max(0.1, node.compute_tflops) * 1e12 * 0.62
        eff_mem_bw = node.mem_bw_gbps * 1e9 * 0.78
        comp_t = (float(prompt_tokens) * n_l * 2.0 * params_per_layer) / eff_flops
        mem_t = (n_l * per_layer_weight_bytes) / eff_mem_bw
        ttft_sec += max(comp_t, mem_t)
        if idx > 0:
            bw_bytes_s = max(1e5, node.bandwidth_mbps * 1_000_000.0 / 8.0)
            ttft_sec += 2.0 * (node.rtt_ms / 1000.0) + (prefill_act_bytes / bw_bytes_s)

    return decode_tok_s, ttft_sec


def _find_fallback_suggestion(
    nodes: list[NodeSpec],
    model: str,
    n_ctx: int,
    os_headroom_gib: float,
) -> str:
    # Check lower quantizations of same model, then smaller models
    candidates = [
        (model, "Q5_K_M", n_ctx),
        (model, "Q4_K_M", n_ctx),
        (model, "Q4_K_M", max(2048, n_ctx // 2)),
        ("30b", "Q4_K_M", 4096),
        ("14b", "Q8_0", 4096),
        ("14b", "Q4_K_M", 4096),
        ("4b", "Q4_K_M", 4096),
    ]
    coord_list = [n for n in nodes if n.role == "coordinator"] or [nodes[0]]
    workers = [n for n in nodes if n is not coord_list[0]]
    ordered = [coord_list[0]] + workers

    for m_key, q_name, ctx_try in candidates:
        fp = estimate_model_memory(m_key, quant=q_name, n_ctx=ctx_try)
        counts = _allocate_layers_on_subset(ordered, fp, os_headroom_gib)
        if counts is not None:
            return (
                f"Cluster fits '{m_key}' at {q_name} (n_ctx={ctx_try}, "
                f"requires {fp.total_required_gib:.2f} GiB)."
            )
    return "Add RAM or reduce OS memory pressure; even 4B Q4_K_M exceeds usable cluster RAM."


def plan_cluster(
    nodes: list[NodeSpec],
    model: str = "30b",
    quant: str = "Q4_K_M",
    n_ctx: int = 4096,
    kv_type: str = "f16",
    os_headroom_gib: float = 2.5,
    prompt_tokens: int = 512,
) -> ClusterPlan:
    """Compute RAM-safe contiguous pipeline split and roofline speed, pruning slow links that hurt tok/s."""
    if not nodes:
        raise ValueError("nodes list cannot be empty")

    spec = get_model_spec(model)
    fp = estimate_model_memory(spec, quant=quant, n_ctx=n_ctx, kv_type=kv_type)

    coords = [n for n in nodes if n.role == "coordinator"]
    coordinator = coords[0] if coords else nodes[0]
    workers = [n for n in nodes if n is not coordinator]
    ordered_all = [coordinator] + workers

    # Compute cluster-wide usable layer budget
    total_layer_cap = 0
    cluster_usable_gib = 0.0
    for idx, node in enumerate(ordered_all):
        usable_gib, _, max_l = _node_layer_capacity(
            node, fp, os_headroom_gib, is_coordinator=(idx == 0)
        )
        cluster_usable_gib += usable_gib
        total_layer_cap += max_l

    all_counts = _allocate_layers_on_subset(ordered_all, fp, os_headroom_gib)
    if all_counts is None:
        missing_layers = max(1, fp.n_layers - total_layer_cap)
        deficit_gib = round(
            max(0.1, missing_layers * fp.per_layer_total_gib),
            2,
        )
        empty_allocs = [
            NodeAllocation(
                name=n.name,
                host=n.host,
                rpc_port=n.rpc_port,
                role="coordinator" if i == 0 else "worker",
                assigned_layers=0,
                layer_range=None,
                used_ram_gib=0.0,
                usable_ram_gib=round(max(0.0, n.ram_gib - os_headroom_gib), 2),
                headroom_gib=round(n.ram_gib, 2),
                tensor_split_weight=0.0,
            )
            for i, n in enumerate(ordered_all)
        ]
        suggestion = _find_fallback_suggestion(ordered_all, spec.key, n_ctx, os_headroom_gib)
        return ClusterPlan(
            fits=False,
            model_key=spec.key,
            display_name=spec.display_name,
            quant=fp.quant,
            n_ctx=n_ctx,
            kv_type=fp.kv_type,
            total_required_gib=round(fp.total_required_gib, 2),
            cluster_usable_gib=round(cluster_usable_gib, 2),
            deficit_gib=deficit_gib,
            allocations=empty_allocs,
            excluded_nodes=[],
            decode_tok_s=0.0,
            decode_bounds_tok_s=(0.0, 0.0),
            ttft_sec=0.0,
            ttft_bounds_sec=(0.0, 0.0),
            warnings=[
                f"OOM REJECTED: {spec.display_name} ({fp.quant}, n_ctx={n_ctx}) needs "
                f"{fp.total_required_gib:.2f} GiB across layers+scratch, exceeding usable "
                f"layer capacity by ~{deficit_gib:.2f} GiB (with {os_headroom_gib:.1f} GiB/node OS headroom)."
            ],
            fallback_suggestion=suggestion,
        )

    # Evaluate all subsets that include coordinator and fit the model to detect slow-node penalties
    fastest_tok_s = -1.0
    fastest_ttft = 0.0
    chosen_subset: list[NodeSpec] = ordered_all
    chosen_counts: list[int] = all_counts
    # Compute naive RAM-proportional split across all nodes (what manual --tensor-split does) for honest comparison
    naive_counts: list[int] = []
    total_raw_ram = sum(n.ram_gib for n in ordered_all)
    rem_naive = fp.n_layers
    for i, n in enumerate(ordered_all):
        _, _, cap_i = _node_layer_capacity(n, fp, os_headroom_gib, is_coordinator=(i == 0))
        share = min(cap_i, max(1 if cap_i > 0 else 0, round(fp.n_layers * (n.ram_gib / total_raw_ram))))
        take_n = min(rem_naive, share)
        naive_counts.append(take_n)
        rem_naive -= take_n
    for i in range(len(ordered_all)):
        if rem_naive <= 0:
            break
        _, _, cap_i = _node_layer_capacity(ordered_all[i], fp, os_headroom_gib, is_coordinator=(i == 0))
        add = min(rem_naive, max(0, cap_i - naive_counts[i]))
        naive_counts[i] += add
        rem_naive -= add
    naive_all_tok_s, _ = _simulate_roofline(
        ordered_all, naive_counts, fp, prompt_tokens=prompt_tokens
    )

    for r in range(0, len(workers) + 1):
        for w_combo in combinations(workers, r):
            sub = [coordinator] + list(w_combo)
            counts = _allocate_layers_on_subset(sub, fp, os_headroom_gib)
            if counts is None:
                continue
            # Prune zero-layer workers from subset evaluation
            active_pairs = [
                (sub[i], counts[i]) for i in range(len(sub)) if i == 0 or counts[i] > 0
            ]
            active_sub = [p[0] for p in active_pairs]
            active_counts = [p[1] for p in active_pairs]
            tok_s, ttft_s = _simulate_roofline(
                active_sub, active_counts, fp, prompt_tokens=prompt_tokens
            )
            # Prefer higher tok/s; break near-ties (within 0.5%) in favor of fewer network hops
            if tok_s > fastest_tok_s * 1.005 or (
                abs(tok_s - fastest_tok_s) <= fastest_tok_s * 0.005
                and len(active_sub) < len(chosen_subset)
            ):
                fastest_tok_s = tok_s
                fastest_ttft = ttft_s
                chosen_subset = active_sub
                chosen_counts = active_counts

    chosen_by_name = {n.name: c for n, c in zip(chosen_subset, chosen_counts)}
    excluded_nodes = [n.name for n in ordered_all if n.name not in chosen_by_name]

    warnings: list[str] = []
    if excluded_nodes:
        for ex_name in excluded_nodes:
            ex_node = [n for n in ordered_all if n.name == ex_name][0]
            warnings.append(
                f"Excluded '{ex_node.name}' ({ex_node.link_type}, RTT={ex_node.rtt_ms:.1f}ms, "
                f"{ex_node.mem_bw_gbps:.0f} GB/s): naive 3-node split drops decode speed from "
                f"{fastest_tok_s:.2f} tok/s down to {naive_all_tok_s:.2f} tok/s."
            )

    for n in chosen_subset[1:]:
        if "wifi" in n.link_type.lower() or n.rtt_ms >= 8.0 or n.bandwidth_mbps < 250.0:
            warnings.append(
                f"Wi-Fi / high-RTT link on '{n.name}' ({n.link_type}, RTT={n.rtt_ms:.1f}ms, "
                f"{n.bandwidth_mbps:.0f} Mbps) is kept only because cluster RAM requires it; "
                "switching this node to Gigabit Ethernet improves TTFT and decode stability."
            )

    if len(chosen_subset) == 1 and len(ordered_all) > 1:
        warnings.append(
            f"Single-node execution on '{coordinator.name}' ({fastest_tok_s:.2f} tok/s) "
            "outperforms multi-node RPC for this model size; zero remote workers needed."
        )

    # Assemble NodeAllocation objects in original node order
    allocations: list[NodeAllocation] = []
    cursor_layer = 0
    active_total_layers = max(1, fp.n_layers)

    for idx, node in enumerate(ordered_all):
        n_l = chosen_by_name.get(node.name, 0)
        is_coord = idx == 0
        usable_gib = max(0.0, node.ram_gib - os_headroom_gib)
        if n_l > 0 or (is_coord and node.name in chosen_by_name):
            fixed_gib = fp.coordinator_fixed_gib if is_coord else fp.worker_scratch_gib
            used_gib = fixed_gib + n_l * fp.per_layer_total_gib
            l_range = (cursor_layer, cursor_layer + n_l - 1) if n_l > 0 else None
            cursor_layer += n_l
            ts_weight = round(n_l / float(active_total_layers), 4)
        else:
            used_gib = 0.0
            l_range = None
            ts_weight = 0.0

        allocations.append(
            NodeAllocation(
                name=node.name,
                host=node.host,
                rpc_port=node.rpc_port,
                role="coordinator" if is_coord else "worker",
                assigned_layers=n_l,
                layer_range=l_range,
                used_ram_gib=round(used_gib, 2),
                usable_ram_gib=round(usable_gib, 2),
                headroom_gib=round(node.ram_gib - used_gib, 2),
                tensor_split_weight=ts_weight,
            )
        )

    # Confidence bounds (+/- 15% accounting for thermal throttling, OS background paging, LAN jitter)
    decode_bounds = (round(fastest_tok_s * 0.85, 2), round(fastest_tok_s * 1.12, 2))
    ttft_bounds = (round(fastest_ttft * 0.88, 2), round(fastest_ttft * 1.20, 2))

    return ClusterPlan(
        fits=True,
        model_key=spec.key,
        display_name=spec.display_name,
        quant=fp.quant,
        n_ctx=n_ctx,
        kv_type=fp.kv_type,
        total_required_gib=round(fp.total_required_gib, 2),
        cluster_usable_gib=round(cluster_usable_gib, 2),
        deficit_gib=0.0,
        allocations=allocations,
        excluded_nodes=excluded_nodes,
        decode_tok_s=round(fastest_tok_s, 2),
        decode_bounds_tok_s=decode_bounds,
        ttft_sec=round(fastest_ttft, 2),
        ttft_bounds_sec=ttft_bounds,
        warnings=warnings,
        fallback_suggestion=None,
    )


def replan_after_drop(
    nodes: list[NodeSpec],
    dropped_names: list[str],
    model: str = "30b",
    quant: str = "Q4_K_M",
    n_ctx: int = 4096,
    kv_type: str = "f16",
    os_headroom_gib: float = 2.5,
) -> dict[str, Any]:
    """Recompute cluster plan when one or more peers drop mid-session."""
    drop_set = {d.strip() for d in dropped_names if d.strip()}
    surviving = [n for n in nodes if n.name not in drop_set]
    if not surviving:
        return {
            "status": "NO_SURVIVING_NODES",
            "dropped": sorted(drop_set),
            "promoted_coordinator": None,
            "plan": None,
            "fallback_plan": None,
            "message": "All cluster nodes dropped; no surviving compute nodes.",
        }

    promoted: str | None = None
    if not any(n.role == "coordinator" for n in surviving):
        # Promote highest-RAM (tie-break highest mem_bw) worker to coordinator
        surviving_sorted = sorted(
            surviving, key=lambda x: (x.ram_gib, x.mem_bw_gbps), reverse=True
        )
        lead = surviving_sorted[0]
        promoted = lead.name
        surviving = [
            NodeSpec(
                name=lead.name,
                host=lead.host,
                rpc_port=lead.rpc_port,
                role="coordinator",
                ram_gib=lead.ram_gib,
                mem_bw_gbps=lead.mem_bw_gbps,
                compute_tflops=lead.compute_tflops,
                link_type=lead.link_type,
                rtt_ms=0.0,
                bandwidth_mbps=lead.bandwidth_mbps,
            )
        ] + surviving_sorted[1:]

    rebalanced = plan_cluster(
        surviving,
        model=model,
        quant=quant,
        n_ctx=n_ctx,
        kv_type=kv_type,
        os_headroom_gib=os_headroom_gib,
    )
    if rebalanced.fits:
        return {
            "status": "REBALANCED",
            "dropped": sorted(drop_set),
            "promoted_coordinator": promoted,
            "plan": rebalanced,
            "fallback_plan": None,
            "message": (
                f"Survived drop of {sorted(drop_set)}: rebalanced {model.upper()} {quant} "
                f"at {rebalanced.decode_tok_s:.2f} tok/s."
            ),
        }

    # Find highest-fidelity runnable fallback plan on surviving nodes
    fallback_candidates = [
        (model, "Q5_K_M", n_ctx),
        (model, "Q4_K_M", n_ctx),
        (model, "Q4_K_M", max(2048, n_ctx // 2)),
        ("30b", "Q4_K_M", 4096),
        ("14b", "Q8_0", 4096),
        ("14b", "Q4_K_M", 4096),
        ("4b", "Q4_K_M", 4096),
    ]
    fallback_plan: ClusterPlan | None = None
    for m_try, q_try, ctx_try in fallback_candidates:
        cand = plan_cluster(
            surviving,
            model=m_try,
            quant=q_try,
            n_ctx=ctx_try,
            kv_type=kv_type,
            os_headroom_gib=os_headroom_gib,
        )
        if cand.fits:
            fallback_plan = cand
            break

    msg = (
        f"Surviving nodes short by {rebalanced.deficit_gib:.2f} GiB for {model.upper()} {quant}. "
        + (
            f"Recommended automatic fallback: {fallback_plan.model_key.upper()} {fallback_plan.quant} "
            f"(n_ctx={fallback_plan.n_ctx}) -> {fallback_plan.decode_tok_s:.2f} tok/s."
            if fallback_plan is not None
            else "No model fits on surviving nodes."
        )
    )
    return {
        "status": "DOWNGRADE_REQUIRED",
        "dropped": sorted(drop_set),
        "promoted_coordinator": promoted,
        "plan": rebalanced,
        "fallback_plan": fallback_plan,
        "message": msg,
    }
