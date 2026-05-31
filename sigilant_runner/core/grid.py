from __future__ import annotations

import os
from typing import List, Tuple, Dict, Optional, Any

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

# Adaptive context ladder used to backfill additional fitting configs.
_CTX_LADDER = [32768, 28672, 24576, 20480, 16384, 12288, 8192, 6144, 4096]

# k8v8 uses half the KV VRAM of k16v16 — apply 0.5× to kv_gb for those combos
_KV_SCALE = {"k16v16": 1.0, "k8v8": 0.5}

# vLLM logical families are not GGUF quants, map to conservative fit factors.
_VLLM_FAMILY_FACTOR: dict[str, float] = {
    "FP16_BASELINE": 1.00,
    "INT8_W8A8": 0.62,
    "AWQ4_MARLIN": 0.34,
    "GPTQ4_MARLIN": 0.34,
}

_FIT_CAP_BY_ENGINE_GPU = {
    "llama.cpp": {
        "l4": 0.86,
        "a10g": 0.88,
        "a100": 0.91,
        "h100": 0.92,
        "default": 0.88,
    },
    "vllm": {
        "l4": 0.80,
        "a10g": 0.82,
        "a100": 0.86,
        "h100": 0.88,
        "default": 0.82,
    },
}


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


def _estimate_for_engine(
    *,
    engine: str,
    quant_label: str,
    context: int,
    kv_type: str,
    model_params_b: float,
) -> float:
    if engine == "vllm":
        # Reuse same estimate pipeline with vLLM-family weight factors.
        fam = quant_label.upper()
        factor = _VLLM_FAMILY_FACTOR.get(fam, 0.60)
        weights = model_params_b * 2 * factor
        kv_raw = (2 * 32 * 128 * context * 2) / (1024 ** 3)
        kv_gb = kv_raw * _KV_SCALE.get(kv_type, 1.0)
        # extra allocator/runtime headroom for vLLM
        return weights + kv_gb + 1.5
    return _vram_estimate_gb(quant_label, context, kv_type, model_params_b)


def _fit_cap(engine: str, hardware_key: str) -> float:
    eng = str(engine or "llama.cpp").lower()
    hw = str(hardware_key or "default").lower()
    table = _FIT_CAP_BY_ENGINE_GPU.get(eng, _FIT_CAP_BY_ENGINE_GPU["llama.cpp"])
    cap = table.get("default", 0.88)
    for k, v in table.items():
        if k == "default":
            continue
        if k in hw:
            cap = v
            break
    env_key = f"SIGILANT_{eng.upper().replace('.', '_')}_FIT_CAP"
    raw = os.environ.get(env_key, "").strip()
    if raw:
        try:
            val = float(raw)
            if 0.50 <= val <= 0.98:
                cap = val
        except Exception:
            pass
    return cap


def _kv_mem_gb_gqa(context: int, kv_type: str, model_profile: Optional[Dict[str, Any]]) -> float:
    mp = model_profile or {}
    n_layers = int(mp.get("n_layers", 32) or 32)
    hidden = int(mp.get("hidden_size", 4096) or 4096)
    n_heads = max(1, int(mp.get("n_heads", 32) or 32))
    n_kv = max(1, int(mp.get("n_kv_heads", n_heads) or n_heads))
    head_dim = int(mp.get("head_dim", 0) or 0)
    if head_dim <= 0:
        head_dim = max(1, hidden // n_heads)
    # 2 (K+V) * layers * ctx * kv_heads * head_dim * fp16_bytes
    kv_raw = (2 * n_layers * context * n_kv * head_dim * 2) / (1024 ** 3)
    return kv_raw * _KV_SCALE.get(kv_type, 1.0)


def _estimate_total_gb(
    *,
    engine: str,
    quant_label: str,
    context: int,
    kv_type: str,
    model_params_b: float,
    model_profile: Optional[Dict[str, Any]],
) -> float:
    fam = str(quant_label or "").upper()
    if engine == "vllm":
        factor = _VLLM_FAMILY_FACTOR.get(fam, 0.60)
    else:
        factor = _QUANT_VRAM_FACTOR.get(fam, 0.60)
    weights = model_params_b * 2 * factor
    kv_gb = _kv_mem_gb_gqa(context, kv_type, model_profile)
    overhead = 1.5 if engine == "vllm" else 0.6
    if bool((model_profile or {}).get("is_moe", False)):
        # MoE runtime safety margin only (router/activation/transient overhead).
        # This is not re-counting expert weights.
        weights *= 1.12
        overhead += 0.8
    return weights + kv_gb + overhead


def _fits(
    *,
    engine: str,
    quant_label: str,
    context: int,
    kv_type: str,
    model_params_b: float,
    vram_gb: float,
    model_profile: Optional[Dict[str, Any]] = None,
    hardware_key: str = "default",
) -> bool:
    fit_cap = vram_gb * _fit_cap(engine, hardware_key)
    est = _estimate_total_gb(
        engine=engine,
        quant_label=quant_label,
        context=context,
        kv_type=kv_type,
        model_params_b=model_params_b,
        model_profile=model_profile,
    )
    return est <= fit_cap


def _best_fitting_ctx(
    *,
    engine: str,
    quant_label: str,
    target_ctx: int,
    kv_type: str,
    model_params_b: float,
    vram_gb: float,
    model_profile: Optional[Dict[str, Any]] = None,
    hardware_key: str = "default",
) -> Optional[int]:
    # Choose highest context <= target that fits.
    for ctx in _CTX_LADDER:
        if ctx > target_ctx:
            continue
        if _fits(
            engine=engine,
            quant_label=quant_label,
            context=ctx,
                kv_type=kv_type,
                model_params_b=model_params_b,
                vram_gb=vram_gb,
                model_profile=model_profile,
                hardware_key=hardware_key,
            ):
            return ctx
    return None


def generate_grid(
    models: List[Tuple[str, str]],  # [(quant_label, path_or_filename), ...]
    vram_gb: float,
    model_params_b: float = 7.0,
    max_configs: int = 16,
    model_repo: str = "",
    engine: str = "llama.cpp",
    model_profile: Optional[Dict[str, Any]] = None,
    hardware_key: str = "default",
) -> List[RunConfig]:
    """Return up to max_configs fit-aware RunConfig objects."""
    configs: List[RunConfig] = []
    seen: set = set()
    allowed_kv = None
    if engine == "vllm":
        raw_kv = os.environ.get("SIGILANT_VLLM_KV_TYPES", "").strip()
        if raw_kv:
            vals = {x.strip().lower() for x in raw_kv.split(",") if x.strip()}
            if vals:
                allowed_kv = vals

    # Pass 1: fill canonical 4 slots per quant with adaptive context backfill.
    for quant_label, model_path in models:
        for ctx, kv_type, regime in _COMBOS:
            if allowed_kv is not None and kv_type.lower() not in allowed_kv:
                continue
            fit_ctx = _best_fitting_ctx(
                engine=engine,
                quant_label=quant_label,
                target_ctx=ctx,
                kv_type=kv_type,
                model_params_b=model_params_b,
                vram_gb=vram_gb,
                model_profile=model_profile,
                hardware_key=hardware_key,
            )
            if fit_ctx is None:
                continue
            sig = (quant_label.upper(), fit_ctx, kv_type)
            if sig in seen:
                continue
            seen.add(sig)
            configs.append(RunConfig(
                quant_label=quant_label,
                context=fit_ctx,
                batch=_BATCH,
                kv_type=kv_type,
                regime=regime,
                model_path=model_path if not model_repo else "",
                model_repo=model_repo,
                model_filename=model_path if model_repo else "",
            ))

    # Pass 2: if still short, backfill additional low-context fits (bias lower-memory KV).
    if len(configs) < max_configs:
        preferred_kv = ["k8v8", "k16v16"]
        for quant_label, model_path in models:
            if len(configs) >= max_configs:
                break
            kv_candidates = [
                kv for kv in preferred_kv
                if allowed_kv is None or kv in allowed_kv
            ]
            for kv_type in kv_candidates:
                for ctx in [6144, 4096]:
                    if len(configs) >= max_configs:
                        break
                    if not _fits(
                        engine=engine,
                        quant_label=quant_label,
                        context=ctx,
                        kv_type=kv_type,
                        model_params_b=model_params_b,
                        vram_gb=vram_gb,
                        model_profile=model_profile,
                        hardware_key=hardware_key,
                    ):
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
                        regime="default",
                        model_path=model_path if not model_repo else "",
                        model_repo=model_repo,
                        model_filename=model_path if model_repo else "",
                    ))

    # Sort: default regime first, then long; within regime larger ctx first
    regime_order = {"default": 0, "long": 1}
    configs.sort(key=lambda c: (regime_order.get(c.regime, 9), -c.context))
    return configs[:max_configs]
