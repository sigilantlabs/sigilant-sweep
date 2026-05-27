from __future__ import annotations

from pathlib import Path

_FALLBACK_PPL_CORPUS = (
    "The transformer architecture has revolutionized natural language processing. "
    "Self-attention mechanisms allow models to weigh the importance of different words "
    "in a sequence when producing representations. Large language models trained on "
    "diverse corpora demonstrate emergent capabilities including in-context learning, "
    "chain-of-thought reasoning, and instruction following. Quantization reduces model "
    "precision to decrease memory footprint, with quality loss scaling as bit-width "
    "decreases. The trade-off between inference speed and output fidelity depends "
    "on model architecture, quantization scheme, and deployment context window."
)


def load_shared_ppl_corpus() -> str:
    """Load the canonical PPL corpus used across engines/backends."""
    candidates = [
        Path("prompts/ppl_corpus_250.txt"),
        Path(__file__).resolve().parents[2] / "prompts" / "ppl_corpus_250.txt",
    ]
    for p in candidates:
        try:
            if p.exists():
                txt = p.read_text(encoding="utf-8").strip()
                if txt:
                    return txt
        except Exception:
            continue
    return _FALLBACK_PPL_CORPUS

