from __future__ import annotations

from typing import List

from .metrics import RunResult


def ranking_key(r: RunResult):
    score = float(r.score or 0.0)
    ttft = float(r.ttft_ms or 1e12)
    tps = float(r.tps or 0.0)
    ppl = float(r.ppl) if r.ppl is not None else 1e9
    label = r.config.label()
    # Deterministic ordering: score desc, ttft asc, tps desc, ppl asc, label asc
    return (-score, ttft, -tps, ppl, label)


def ranked_succeeded(results: List[RunResult]) -> List[RunResult]:
    ok = [r for r in results if r.succeeded and r.score is not None]
    return sorted(ok, key=ranking_key)

