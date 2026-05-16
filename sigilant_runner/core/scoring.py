from __future__ import annotations

from typing import List, Tuple

from .metrics import RunResult


def resolve_weight_profile(profile: str) -> Tuple[float, float, float]:
    p = (profile or "balanced").strip().lower()
    if p == "latency":
        return 0.50, 0.30, 0.20
    if p == "quality":
        return 0.30, 0.20, 0.50
    return 0.40, 0.20, 0.40


def compute_scores(results: List[RunResult], profile: str = "balanced") -> List[RunResult]:
    """Normalise and compute Sigilant Score for each succeeded result.

    Weights: 40% TPS, 20% TTFT (lower is better), 40% PPL quality (lower PPL is better).
    When PPL is unavailable (vLLM), the weight shifts: 60% TPS, 40% TTFT.
    Score is expressed as 0–100.
    """
    valid = [r for r in results if r.succeeded]
    if not valid:
        return results

    has_ppl = any(r.ppl is not None for r in valid)

    max_tps   = max((r.tps for r in valid), default=1.0)
    has_tail_tps = any((r.tps_p95 is not None and r.tps_p95 > 0) for r in valid)
    max_tps_p95 = max((r.tps_p95 for r in valid if r.tps_p95 and r.tps_p95 > 0), default=max_tps)
    min_ttft  = min((r.ttft_ms for r in valid if r.ttft_ms > 0), default=1.0)
    has_tail_ttft = any((r.ttft_p95_ms is not None and r.ttft_p95_ms > 0) for r in valid)
    min_ttft_p95 = min((r.ttft_p95_ms for r in valid if r.ttft_p95_ms and r.ttft_p95_ms > 0), default=min_ttft)

    if has_ppl:
        ppl_valid = [r.ppl for r in valid if r.ppl is not None and r.ppl > 0]
        min_ppl   = min(ppl_valid) if ppl_valid else None

    w_tps, w_ttft, w_ppl = resolve_weight_profile(profile)

    for r in results:
        if not r.succeeded:
            continue

        tps_p50_score  = r.tps / max_tps if max_tps > 0 else 0.0
        if has_tail_tps and r.tps_p95 and r.tps_p95 > 0:
            tps_p95_score = r.tps_p95 / max_tps_p95 if max_tps_p95 > 0 else 0.0
            # Symmetric tail-aware blend with TTFT.
            tps_score = 0.5 * tps_p50_score + 0.5 * tps_p95_score
        else:
            tps_score = tps_p50_score
        ttft_p50_score = min_ttft / r.ttft_ms if r.ttft_ms > 0 else 0.0
        if has_tail_ttft and r.ttft_p95_ms and r.ttft_p95_ms > 0:
            ttft_p95_score = min_ttft_p95 / r.ttft_p95_ms
            # Tail-aware blend when high-trial stats are available.
            ttft_score = 0.5 * ttft_p50_score + 0.5 * ttft_p95_score
        else:
            ttft_score = ttft_p50_score
        r.tps_norm = round(tps_score * 100, 1)
        r.ttft_norm = round(ttft_score * 100, 1)

        if has_ppl and r.ppl is not None:
            # Symmetric normalization with TTFT: lower-is-better ratio to best.
            ppl_score = (min_ppl / r.ppl) if (min_ppl and r.ppl > 0) else 0.0
            r.ppl_norm = round(ppl_score * 100, 1)
            r.score = round((w_tps * tps_score + w_ttft * ttft_score + w_ppl * ppl_score) * 100)
        else:
            r.ppl_norm = None
            # Renormalize TPS/TTFT when PPL is unavailable.
            perf_sum = max(1e-9, (w_tps + w_ttft))
            r.score = round((((w_tps / perf_sum) * tps_score) + ((w_ttft / perf_sum) * ttft_score)) * 100)

    return results
