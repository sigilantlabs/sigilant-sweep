from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .metrics import RunResult
from .ranking import ranked_succeeded


@dataclass
class StabilityReport:
    winner_label: str
    winner_score: float
    runner_up_label: Optional[str]
    runner_up_score: Optional[float]
    top2_gap_abs: float
    top2_gap_pct: float
    confidence: str
    notes: str


def build_stability_report(results: List[RunResult]) -> Optional[StabilityReport]:
    ok = ranked_succeeded(results)
    if not ok:
        return None
    w = ok[0]
    r2 = ok[1] if len(ok) > 1 else None
    ws = float(w.score or 0.0)
    rs = float(r2.score or 0.0) if r2 is not None and r2.score is not None else 0.0
    gap = max(0.0, ws - rs)
    gap_pct = (gap / ws * 100.0) if ws > 0 else 0.0

    if gap >= 5:
        conf = "high"
        note = "Winner separated clearly from runner-up."
    elif gap >= 2:
        conf = "medium"
        note = "Winner has moderate margin; repeat run recommended for audit."
    else:
        conf = "low"
        note = "Near-tie winner; increase trials or run top-2 replay."

    return StabilityReport(
        winner_label=w.config.label(),
        winner_score=ws,
        runner_up_label=(r2.config.label() if r2 else None),
        runner_up_score=(float(r2.score) if r2 and r2.score is not None else None),
        top2_gap_abs=round(gap, 3),
        top2_gap_pct=round(gap_pct, 2),
        confidence=conf,
        notes=note,
    )


def to_dict(rep: Optional[StabilityReport]) -> Optional[Dict[str, Any]]:
    if rep is None:
        return None
    return {
        "winner_label": rep.winner_label,
        "winner_score": rep.winner_score,
        "runner_up_label": rep.runner_up_label,
        "runner_up_score": rep.runner_up_score,
        "top2_gap_abs": rep.top2_gap_abs,
        "top2_gap_pct": rep.top2_gap_pct,
        "confidence": rep.confidence,
        "notes": rep.notes,
    }
