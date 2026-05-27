from __future__ import annotations

from pathlib import Path

_FALLBACK_EVAL_PROMPT = (
    "Explain the key architectural differences between transformer encoder and decoder models. "
    "Include details about attention mechanisms, typical use cases, and how self-attention "
    "differs from cross-attention."
)


def load_default_eval_prompt() -> str:
    candidates = [
        Path("prompts/default_eval_prompt.txt"),
        Path(__file__).resolve().parents[2] / "prompts" / "default_eval_prompt.txt",
    ]
    for p in candidates:
        try:
            if p.exists():
                txt = p.read_text(encoding="utf-8").strip()
                if txt:
                    return txt
        except Exception:
            continue
    return _FALLBACK_EVAL_PROMPT

