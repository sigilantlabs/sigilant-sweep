"""Model resolution — HuggingFace repo or local GGUF path."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

# Quant files to look for, in preference order.
# 3-bit policy: prefer IQ3_M; use Q3_K_M only as fallback when IQ3_M is absent.
_TARGET_QUANTS = ["Q8_0", "Q5_K_M", "Q4_K_M", "IQ3_M"]

_CACHE_DIR = Path.home() / ".cache" / "sigilant" / "models"


def resolve(model: str, engine: str, backend: str = "local") -> Tuple[List[Tuple[str, str]], str, str]:
    """Resolve a model identifier to a list of (quant_label, path_or_filename) tuples.

    Returns:
        models    — list of (quant_label, path_or_filename)
        label     — display name for the header
        repo_id   — HF repo ID if remote, "" if local
    """
    if engine == "vllm":
        label = model.split("/")[-1] if "/" in model else Path(model).stem
        # Dedicated vLLM families (separate from llama.cpp GGUF quants):
        # 4 families × 4 ctx/kv combos = 16 configs.
        families = ["FP16_BASELINE", "INT8_W8A8", "AWQ4_MARLIN", "GPTQ4_MARLIN"]
        raw = os.environ.get("SIGILANT_VLLM_FAMILIES", "").strip()
        if raw:
            wanted = {x.strip().upper() for x in raw.split(",") if x.strip()}
            filtered = [f for f in families if f in wanted]
            if filtered:
                families = filtered
        return [(fam, model) for fam in families], label, model if "/" in model else ""

    p = Path(model)
    if p.exists() and p.suffix.lower() == ".gguf":
        quant = _quant_from_name(p.name)
        return [(quant, str(p))], p.stem, ""

    # HuggingFace repo ID
    if backend in ("modal", "runpod"):
        # Remote backends: list files only — download happens inside the container
        return _list_from_hub(model)
    return _fetch_from_hub(model)


def _list_from_hub(repo_id: str) -> Tuple[List[Tuple[str, str]], str, str]:
    """List GGUF files in a HF repo without downloading anything locally."""
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        raise RuntimeError(
            "huggingface-hub is required to list models.\n"
            "  pip install 'sigilant-runner[modal]'"
        )

    all_files = [f for f in list_repo_files(repo_id) if f.lower().endswith(".gguf")]
    if not all_files:
        raise ValueError(f"No .gguf files found in {repo_id}")

    selected: List[Tuple[str, str]] = []
    for target in _TARGET_QUANTS:
        for fname in all_files:
            if target.lower() in fname.lower():
                selected.append((target, fname))
                break

    # 3-bit fallback when IQ3_M is unavailable.
    has_iq3 = any(q == "IQ3_M" for q, _ in selected)
    if not has_iq3:
        for fname in all_files:
            if "q3_k_m" in fname.lower():
                selected.append(("Q3_K_M", fname))
                break

    if not selected:
        selected = [(_quant_from_name(f), f) for f in all_files[:4]]

    label = repo_id.split("/")[-1]
    return selected, label, repo_id


def _fetch_from_hub(repo_id: str) -> Tuple[List[Tuple[str, str]], str, str]:
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface-hub is required to download models.\n"
            "  pip install huggingface-hub"
        )

    all_files = [f for f in list_repo_files(repo_id) if f.lower().endswith(".gguf")]
    if not all_files:
        raise ValueError(f"No .gguf files found in {repo_id}")

    selected: List[Tuple[str, str]] = []
    for target in _TARGET_QUANTS:
        for fname in all_files:
            if target.lower() in fname.lower():
                selected.append((target, fname))
                break

    # 3-bit fallback when IQ3_M is unavailable.
    has_iq3 = any(q == "IQ3_M" for q, _ in selected)
    if not has_iq3:
        for fname in all_files:
            if "q3_k_m" in fname.lower():
                selected.append(("Q3_K_M", fname))
                break

    if not selected:
        # Fallback: take up to 4 any .gguf files
        selected = [(_quant_from_name(f), f) for f in all_files[:4]]

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_dir = _CACHE_DIR / repo_id.replace("/", "--")

    result: List[Tuple[str, str]] = []
    for quant_label, filename in selected:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
        )
        result.append((quant_label, local_path))

    label = repo_id.split("/")[-1]
    return result, label, repo_id


def _quant_from_name(filename: str) -> str:
    m = re.search(
        r"(Q\d+_K_[MS]|Q\d+_K|Q\d+_\d|Q\d+|F16|BF16|F32)",
        filename,
        re.IGNORECASE,
    )
    return m.group(1).upper() if m else "UNKNOWN"


def infer_params_b(model: str) -> float:
    """Infer parameter count (billions) from a model name or repo ID.

    Handles: Mistral-7B, Qwen2.5-14B, Phi-3-mini-3.8B, Llama-3-70B, 0.5B, etc.
    Returns 7.0 if nothing can be inferred.
    """
    m = re.search(r"[-_/](\d+\.?\d*)[bB](?:[-_\s]|$)", model)
    if not m:
        m = re.search(r"(\d+\.?\d*)[bB](?:[-_\s]|$)", model)
    return float(m.group(1)) if m else 7.0
