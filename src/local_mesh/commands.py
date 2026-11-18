# generated-by: gemini-ultra
"""Upstream `llama.cpp` `rpc-server`, `llama-server`, and `llama-cli` command generator."""

from __future__ import annotations

from typing import Any

from local_mesh.models import get_model_spec
from local_mesh.planner import ClusterPlan

SECURITY_NOTES = [
    "SECURITY WARNING: Upstream llama.cpp `rpc-server` is unauthenticated and unencrypted.",
    "Never expose `rpc-server` on public Wi-Fi, guest networks, or WAN router port-forwards.",
    "Bind `--host` to each worker's isolated home-LAN `.local` interface or tunnel over SSH (`ssh -L 50052:127.0.0.1:50052`) / WireGuard.",
]


def generate_llama_cpp_commands(
    plan: ClusterPlan,
    model_path: str | None = None,
    server_port: int = 8080,
) -> dict[str, Any]:
    """Generate copy-pasteable worker `rpc-server` and coordinator `llama-server`/`llama-cli` commands."""
    spec = get_model_spec(plan.model_key)
    resolved_model_path = model_path or f"./models/{spec.default_gguf_template.format(quant=plan.quant.lower())}"

    if not plan.fits:
        return {
            "worker_commands": [],
            "coordinator_server_cmd": "",
            "coordinator_cli_cmd": "",
            "security_notes": SECURITY_NOTES,
        }

    active_allocs = [a for a in plan.allocations if a.role == "coordinator" or a.assigned_layers > 0]
    worker_allocs = [a for a in active_allocs if a.role == "worker" and a.assigned_layers > 0]

    worker_commands: list[dict[str, Any]] = []
    rpc_endpoints: list[str] = []
    for w in worker_allocs:
        worker_commands.append(
            {
                "node": w.name,
                "host": w.host,
                "port": w.rpc_port,
                "assigned_layers": w.assigned_layers,
                "used_ram_gib": w.used_ram_gib,
                "command": f"rpc-server --host {w.host} -p {w.rpc_port}",
            }
        )
        rpc_endpoints.append(f"{w.host}:{w.rpc_port}")

    tensor_ratios = ",".join(str(a.assigned_layers) for a in active_allocs)

    if rpc_endpoints:
        rpc_joined = ",".join(rpc_endpoints)
        coord_server = (
            f"llama-server -m {resolved_model_path} --host 127.0.0.1 --port {server_port} "
            f"-c {plan.n_ctx} -ngl 99 --split-mode layer "
            f"--rpc {rpc_joined} --tensor-split {tensor_ratios}"
        )
        coord_cli = (
            f"llama-cli -m {resolved_model_path} -c {plan.n_ctx} -ngl 99 --split-mode layer "
            f"--rpc {rpc_joined} --tensor-split {tensor_ratios} -p \"Explain pipeline parallelism in 3 sentences.\""
        )
    else:
        coord_server = (
            f"llama-server -m {resolved_model_path} --host 127.0.0.1 --port {server_port} "
            f"-c {plan.n_ctx} -ngl 99"
        )
        coord_cli = (
            f"llama-cli -m {resolved_model_path} -c {plan.n_ctx} -ngl 99 "
            f"-p \"Explain pipeline parallelism in 3 sentences.\""
        )

    return {
        "worker_commands": worker_commands,
        "coordinator_server_cmd": coord_server,
        "coordinator_cli_cmd": coord_cli,
        "security_notes": SECURITY_NOTES,
    }
