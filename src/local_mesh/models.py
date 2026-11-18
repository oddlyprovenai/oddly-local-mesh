# generated-by: gemini-ultra
"""GGUF parameter, quantization bits-per-weight, KV-cache, and runtime scratch byte formulas."""

from __future__ import annotations

from dataclasses import dataclass

GIB = float(1024**3)

QUANT_BITS_PER_WEIGHT: dict[str, float] = {
    "Q4_K_M": 4.85,
    "Q5_K_M": 5.70,
    "Q8_0": 8.50,
    "F16": 16.00,
}

KV_BYTES_PER_ELEMENT: dict[str, float] = {
    "f16": 2.0,
    "q8_0": 1.0625,
    "q4_0": 0.5625,
}


@dataclass(frozen=True)
class ModelSpec:
    """Architectural parameters of a dense GGUF language model."""

    key: str
    display_name: str
    params_b: float
    n_layers: int
    hidden_dim: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    default_gguf_template: str


MODEL_CATALOG: dict[str, ModelSpec] = {
    "4b": ModelSpec(
        key="4b",
        display_name="Qwen3.5-4B (Dense)",
        params_b=4.0,
        n_layers=36,
        hidden_dim=2560,
        n_kv_heads=8,
        head_dim=128,
        vocab_size=151936,
        default_gguf_template="qwen3.5-4b-{quant}.gguf",
    ),
    "14b": ModelSpec(
        key="14b",
        display_name="Qwen2.5-14B (Dense)",
        params_b=14.7,
        n_layers=48,
        hidden_dim=5120,
        n_kv_heads=8,
        head_dim=128,
        vocab_size=152064,
        default_gguf_template="qwen2.5-14b-instruct-{quant}.gguf",
    ),
    "30b": ModelSpec(
        key="30b",
        display_name="Qwen2.5-32B / 30B-Class (Dense)",
        params_b=32.5,
        n_layers=64,
        hidden_dim=5120,
        n_kv_heads=8,
        head_dim=128,
        vocab_size=152064,
        default_gguf_template="qwen2.5-32b-instruct-{quant}.gguf",
    ),
    "70b": ModelSpec(
        key="70b",
        display_name="Llama-3.3-70B (Dense)",
        params_b=70.6,
        n_layers=80,
        hidden_dim=8192,
        n_kv_heads=8,
        head_dim=128,
        vocab_size=128256,
        default_gguf_template="llama-3.3-70b-instruct-{quant}.gguf",
    ),
}

_MODEL_ALIASES: dict[str, str] = {
    "4b": "4b",
    "qwen3.5-4b": "4b",
    "14b": "14b",
    "qwen2.5-14b": "14b",
    "30b": "30b",
    "32b": "30b",
    "qwen2.5-32b": "30b",
    "llama-30b": "30b",
    "70b": "70b",
    "llama-3.3-70b": "70b",
}


@dataclass(frozen=True)
class MemoryFootprint:
    """Computed GGUF + KV-cache + runtime buffer memory breakdown in GiB."""

    model_key: str
    display_name: str
    quant: str
    n_ctx: int
    kv_type: str
    n_layers: int
    hidden_dim: int
    params_b: float
    total_weight_gib: float
    embedding_and_head_gib: float
    coordinator_scratch_gib: float
    coordinator_fixed_gib: float
    worker_scratch_gib: float
    per_layer_weight_gib: float
    per_layer_kv_gib: float
    per_layer_total_gib: float
    total_kv_gib: float
    total_required_gib: float


def get_model_spec(model_name: str | ModelSpec) -> ModelSpec:
    """Lookup a ModelSpec by short key (`4b`, `14b`, `30b`, `70b`) or alias."""
    if isinstance(model_name, ModelSpec):
        return model_name
    normalized = model_name.strip().lower()
    canonical = _MODEL_ALIASES.get(normalized)
    if canonical is None:
        raise ValueError(
            f"Unknown model {model_name!r}; supported keys: {sorted(MODEL_CATALOG.keys())}"
        )
    return MODEL_CATALOG[canonical]


def estimate_model_memory(
    model: str | ModelSpec,
    quant: str = "Q4_K_M",
    n_ctx: int = 4096,
    kv_type: str = "f16",
) -> MemoryFootprint:
    """Compute GGUF weight bytes, KV-cache bytes, and llama.cpp coordinator/worker scratch buffers."""
    spec = get_model_spec(model)
    q_norm = quant.strip().upper()
    if q_norm not in QUANT_BITS_PER_WEIGHT:
        raise ValueError(
            f"Unsupported quantization {quant!r}; expected {list(QUANT_BITS_PER_WEIGHT)}"
        )

    kv_norm = kv_type.strip().lower()
    if kv_norm not in KV_BYTES_PER_ELEMENT:
        raise ValueError(
            f"Unsupported kv_type {kv_type!r}; expected {list(KV_BYTES_PER_ELEMENT)}"
        )
    if n_ctx < 256:
        raise ValueError("n_ctx must be >= 256")

    bpw = QUANT_BITS_PER_WEIGHT[q_norm]
    total_weight_bytes = spec.params_b * 1e9 * (bpw / 8.0)

    # Token embedding + output LM head remain slightly higher precision in K-quants (~6.0 bpw minimum)
    emb_bpw = max(bpw, 6.0) if q_norm != "F16" else 16.0
    emb_and_head_bytes = 2.0 * spec.vocab_size * spec.hidden_dim * (emb_bpw / 8.0)
    emb_and_head_bytes = min(emb_and_head_bytes, 0.20 * total_weight_bytes)

    layer_pool_bytes = max(0.0, total_weight_bytes - emb_and_head_bytes)
    per_layer_weight_bytes = layer_pool_bytes / float(spec.n_layers)

    # KV cache per layer: 2 (K and V) * n_ctx * n_kv_heads * head_dim * bytes_per_elem
    kv_elem_bytes = KV_BYTES_PER_ELEMENT[kv_norm]
    per_layer_kv_bytes = (
        2.0 * float(n_ctx) * float(spec.n_kv_heads) * float(spec.head_dim) * kv_elem_bytes
    )

    # llama.cpp compute graph + activation staging scratch scales with hidden_dim
    coordinator_scratch_gib = round(0.40 + 0.15 * (spec.hidden_dim / 1024.0), 3)
    worker_scratch_gib = round(0.20 + 0.05 * (spec.hidden_dim / 1024.0), 3)

    emb_gib = emb_and_head_bytes / GIB
    coord_fixed_gib = emb_gib + coordinator_scratch_gib
    per_layer_w_gib = per_layer_weight_bytes / GIB
    per_layer_kv_gib = per_layer_kv_bytes / GIB
    per_layer_total_gib = per_layer_w_gib + per_layer_kv_gib
    total_kv_gib = per_layer_kv_gib * spec.n_layers
    total_weight_gib = total_weight_bytes / GIB
    total_required_gib = coord_fixed_gib + (per_layer_total_gib * spec.n_layers)

    return MemoryFootprint(
        model_key=spec.key,
        display_name=spec.display_name,
        quant=q_norm,
        n_ctx=n_ctx,
        kv_type=kv_norm,
        n_layers=spec.n_layers,
        hidden_dim=spec.hidden_dim,
        params_b=spec.params_b,
        total_weight_gib=round(total_weight_gib, 4),
        embedding_and_head_gib=round(emb_gib, 4),
        coordinator_scratch_gib=coordinator_scratch_gib,
        coordinator_fixed_gib=round(coord_fixed_gib, 4),
        worker_scratch_gib=worker_scratch_gib,
        per_layer_weight_gib=round(per_layer_w_gib, 6),
        per_layer_kv_gib=round(per_layer_kv_gib, 6),
        per_layer_total_gib=round(per_layer_total_gib, 6),
        total_kv_gib=round(total_kv_gib, 4),
        total_required_gib=round(total_required_gib, 4),
    )
