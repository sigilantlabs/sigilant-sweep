from __future__ import annotations

from typing import List, Tuple

from .metrics import RunConfig

# VRAM multipliers relative to fp16 baseline (model_params_b × 2 bytes/param)
_QUANT_VRAM_FACTOR: dict[str, float] = {
    "IQ3_M":   0.38,
    "Q3_K_M":  0.38,
    "Q4_0":    0.45,
    "Q4_K_M":  0.48,
    "Q5_0":    0.55,
    "Q5_K_M":  0.58,
    "Q6_K":    0.68,
    "Q8_0":    0.88,
    "F16":     1.00,
    "BF16":    1.00,
    "UNKNOWN": 0.60,
}

# CTX ladder — matches autotune regimes (default=8192, long=16384, xl=32768)
_CTX_DEFAULT = 8192
_CTX_LONG    = 16384
_CTX_XL      = 32768

# Batch is always 4 — matches autotune kv_fixed_batch
_BATCH = 4

# 4 (ctx, kv_type, regime) combos per quant — mirrors autotune grid exactly:
#   (default ctx, k16v16)  → default regime, full-precision KV
#   (long ctx,   k8v8)     → default regime, KV-compressed (saves ~50% KV VRAM → bigger ctx)
#   (long ctx,   k16v16)   → long regime, full-precision KV
#   (xl ctx,     k8v8)     → long regime, KV-compressed
_COMBOS = [
    (_CTX_DEFAULT, "k16v16", "default"),
    (_CTX_LONG,    "k8v8",   "default"),
    (_CTX_LONG,    "k16v16", "long"),
    (_CTX_XL,      "k8v8",   "long"),
]

# k8v8 uses half the KV VRAM of k16v16 — apply 0.5× to kv_gb for those combos
_KV_SCALE = {"k16v16": 1.0, "k8v8": 0.5}


def _vram_estimate_gb(
    quant_label: str,
    context: int,
    kv_type: str,
    model_params_b: float,
) -> float:
    factor   = _QUANT_VRAM_FACTOR.get(quant_label.upper(), 0.60)
    weights  = model_params_b * 2 * factor
    # KV: 2 (K+V) × 32 layers × 128 head_dim × ctx × dtype_bytes (fp16=2)
    kv_raw   = (2 * 32 * 128 * context * 2) / (1024 ** 3)
    kv_gb    = kv_raw * _KV_SCALE.get(kv_type, 1.0)
    return weights + kv_gb


def generate_grid(
    models: List[Tuple[str, str]],  # [(quant_label, path_or_filename), ...]
    vram_gb: float,
    model_params_b: float = 7.0,
    max_configs: int = 16,
    model_repo: str = "",
    engine: str = "llama.cpp",
) -> List[RunConfig]:
    """Return up to max_configs RunConfig objects that fit within vram_gb."""
    configs: List[RunConfig] = []
    seen: set = set()
    for quant_label, model_path in models:
        for ctx, kv_type, regime in _COMBOS:
            est = _vram_estimate_gb(quant_label, ctx, kv_type, model_params_b)
            if est > vram_gb * 0.90:
                continue
            sig = (quant_label.upper(), ctx, kv_type)
            if sig in seen:
                continue
            seen.add(sig)
            configs.append(RunConfig(
                quant_label=quant_label,
                context=ctx,
                batch=_BATCH,
                kv_type=kv_type,
                regime=regime,
                model_path=model_path if not model_repo else "",
                model_repo=model_repo,
                model_filename=model_path if model_repo else "",
            ))

    # Sort: default regime first, then long; within regime larger ctx first
    regime_order = {"default": 0, "long": 1}
    configs.sort(key=lambda c: (regime_order.get(c.regime, 9), -c.context))
    return configs[:max_configs]
