from __future__ import annotations

import json
import math
import shlex
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..__init__ import __version__
from ..core.metrics import RunResult
from ..core.diagnostics import StabilityReport, to_dict as stability_to_dict
from ..core.ranking import ranked_succeeded

_DEFAULT_PATH = "sigilant_results.json"


def _fmt_num(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return str(value)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return str(value)


def export_json(results: List[RunResult], path: str = _DEFAULT_PATH) -> str:
    best = _best_result_from_results(results)
    payload = {
        "schema":  "sigilant.runner.v0.1",
        "version": __version__,
        "best_config": _result_to_dict(best) if best is not None else None,
        "results": [_result_to_dict(r) for r in results],
    }
    Path(path).write_text(json.dumps(payload, indent=2))
    return path


def export_bundle(
    *,
    results: List[RunResult],
    path_json: str,
    path_md: str,
    path_svg: str,
    path_terminal_txt: str,
    context: Dict[str, Any],
    stability: Optional[StabilityReport],
    baseline_compare: Optional[Dict[str, Any]] = None,
    agent_smoke: Optional[Dict[str, Any]] = None,
    confidence_inputs: Optional[Dict[str, Any]] = None,
    depth_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    best = _best_result_from_results(results)
    payload = {
        "schema": "sigilant.runner.v0.2",
        "version": __version__,
        "context": context,
        "stability": stability_to_dict(stability),
        "baseline_compare": baseline_compare,
        "agent_smoke": agent_smoke,
        "confidence_inputs": confidence_inputs,
        "depth_profile": depth_profile,
        "best_config": _result_to_dict(best) if best is not None else None,
        "results": [_result_to_dict(r) for r in results],
    }
    Path(path_json).write_text(json.dumps(payload, indent=2))
    Path(path_md).write_text(_build_md(payload))
    Path(path_svg).write_text(_build_frontier_svg(results, context=context))
    Path(path_terminal_txt).write_text(_build_terminal_like(payload))
    return {"json": path_json, "md": path_md, "svg": path_svg, "terminal": path_terminal_txt}


def _result_to_dict(r: RunResult) -> dict:
    status = "pass"
    if not r.succeeded:
        err = str(r.error or "").lower()
        if err.startswith("skipped_context_overflow"):
            status = "skipped_context_overflow"
        elif err.startswith("skipped_capacity_limit"):
            status = "skipped_capacity_limit"
        elif "startup_failed" in err and "free memory on device" in err:
            status = "failed_startup_oom"
        elif "readtimeout" in err or "timed out" in err:
            status = "failed_runtime_timeout"
        else:
            status = "failed_runtime"
    return {
        "config": {
            "quant_label":    r.config.quant_label,
            "context":        r.config.context,
            "batch":          r.config.batch,
            "kv_type":        r.config.kv_type,
            "depth_label":    r.config.depth_label or None,
            "actual_kv_dtype": (r.preflight or {}).get("actual_kv_dtype"),
            "label":          r.config.label(),
        },
        "tps":      r.tps,
        "tps_p95":  r.tps_p95,
        "ttft_ms":  r.ttft_ms,
        "ttft_p95_ms": r.ttft_p95_ms,
        "itl_ms":   r.itl_ms,
        "ppl":      r.ppl,
        "tps_norm": r.tps_norm,
        "ttft_norm": r.ttft_norm,
        "ppl_norm": r.ppl_norm,
        "score":    r.score,
        "status":   status,
        "error":    r.error,
        "preflight": r.preflight,
    }


def _best_result_from_results(results: List[RunResult]) -> Optional[RunResult]:
    ok = [r for r in results if r.succeeded]
    if not ok:
        return None
    return sorted(
        ok,
        key=lambda x: (
            -float(x.score or 0.0),
            float(x.ttft_ms or 1e12),
            -float(x.tps or 0.0),
            float(x.ppl if x.ppl is not None else 1e9),
            str(x.config.label()),
        ),
    )[0]


def _build_md(payload: Dict[str, Any]) -> str:
    ctx = payload.get("context") or {}
    rows = payload.get("results") or []
    # Deterministic tie-break to match CLI winner logic.
    ok_sorted = sorted(
        [r for r in rows if not r.get("error")],
        key=lambda x: (
            -float(x.get("score") or 0.0),
            float(x.get("ttft_ms") or 1e12),
            -float(x.get("tps") or 0.0),
            float(x.get("ppl") if x.get("ppl") is not None else 1e9),
            str(x.get("config", {}).get("label") or ""),
        ),
    )
    best = ok_sorted[0] if ok_sorted else None
    stab = payload.get("stability") or {}
    repro = ctx.get("repro_command")

    lines = [
        "# Sigilant Runner Summary",
        "",
        f"- Model: `{ctx.get('model')}`",
        f"- Backend/Engine: `{ctx.get('backend')}` / `{ctx.get('engine')}`",
        f"- Hardware: `{ctx.get('hardware')}`",
        f"- Trials: `{ctx.get('trials')}`",
        f"- Score profile: `{ctx.get('score_profile', 'balanced')}`",
        f"- Benchmark mode: `{ctx.get('benchmark_mode', 'ranking')}`",
        "",
    ]
    if ctx.get("benchmark_prompt_source"):
        lines += [
            "## Prompt Provenance",
            f"- Source: `{ctx.get('benchmark_prompt_source')}`",
            f"- Chars: `{ctx.get('benchmark_prompt_chars')}`",
            f"- Tokens (est): `{ctx.get('benchmark_prompt_tokens_est')}`",
            f"- SHA12: `{ctx.get('benchmark_prompt_sha12')}`",
            "",
        ]
    if best:
        lines += [
            "## Winner",
            f"- Config: `{best['config']['label']}`",
            f"- Score: `{best.get('score')}`",
            f"- TPS / TTFT / PPL: `{_fmt_num(best.get('tps'), 1)}` / `{_fmt_num(best.get('ttft_ms'), 1, ' ms')}` / `{_fmt_num(best.get('ppl'), 2)}`",
            "",
        ]
    lines += ["## Quick Wow", f"- Recommendation ready in one run: `{best['config']['label']}`" if best else "- Recommendation unavailable"]
    if stab:
        lines += [
            "- Stability confidence:",
            f"  `{stab.get('confidence')}` (top-2 gap: `{_fmt_num(stab.get('top2_gap_abs'), 2)}` points, `{_fmt_num(stab.get('top2_gap_pct'), 2)}%`)",
            "",
            "## Stability",
            f"- Confidence: `{stab.get('confidence')}`",
            f"- Top-2 gap: `{_fmt_num(stab.get('top2_gap_abs'), 2)}` points (`{_fmt_num(stab.get('top2_gap_pct'), 2)}%` of winner score)",
            f"- Notes: {stab.get('notes')}",
            "",
        ]
    bc = payload.get("baseline_compare")
    if isinstance(bc, dict) and bc.get("found"):
        lines += [
            "## Baseline Compare",
            f"- Baseline: `{bc.get('baseline_label')}`",
            f"- Winner delta score: `{_fmt_num(bc.get('delta_score'), 2)}`",
            f"- Winner delta TPS: `{_fmt_num(bc.get('delta_tps'), 2)}`",
            f"- Winner delta TPS p95: `{_fmt_num(bc.get('delta_tps_p95'), 2)}`",
            f"- Winner delta TTFT(ms): `{_fmt_num(bc.get('delta_ttft_ms'), 1)}`",
            f"- Winner delta TTFT p95(ms): `{_fmt_num(bc.get('delta_ttft_p95_ms'), 1)}`",
            f"- Winner delta PPL: `{_fmt_num(bc.get('delta_ppl'), 2)}`",
            "",
        ]
    ci = payload.get("confidence_inputs")
    if isinstance(ci, dict) and ci:
        lines += [
            "## Confidence Inputs",
            f"- Target: `{ci.get('confidence_target')}`",
            f"- Initial trials: `{ci.get('trials_initial')}`",
            f"- Top-2 gap before: `{_fmt_num(ci.get('gap_abs_before'), 2)}` points (`{_fmt_num(ci.get('gap_pct_before'), 2)}%`)",
            f"- Top-2 variance proxy before: `{_fmt_num(ci.get('variance_pct_before'), 2)}%`",
            f"- Replay triggered: `{ci.get('replay_triggered')}`",
            f"- Replay reason: `{ci.get('replay_reason')}`",
            f"- Replay extra trials: `{ci.get('replay_extra_trials')}`",
            f"- Replay outcome: `{ci.get('replay_outcome')}`",
            f"- Top-2 gap after: `{_fmt_num(ci.get('gap_pct_after'), 2)}%`",
            f"- Top-2 variance proxy after: `{_fmt_num(ci.get('variance_pct_after'), 2)}%`",
            "",
        ]
    smoke = payload.get("agent_smoke")
    if isinstance(smoke, dict):
        lines += [
            "## Agent Smoke",
            f"- Pass rate: `{smoke.get('passed')}/{smoke.get('total')} ({_fmt_pct(smoke.get('pass_rate'))})`",
            f"- Diagnosis: `{smoke.get('diagnosis')}`",
            f"- Status: `{smoke.get('status')}`",
            f"- Failed checks: `{', '.join(smoke.get('failed_checks') or [])}`",
            f"- Harness errors: `{smoke.get('error_count')}` parse-failures: `{smoke.get('parse_fail_count')}`",
            f"- Note: {smoke.get('note')}",
            "",
        ]
    dp = payload.get("depth_profile")
    if isinstance(dp, dict) and (dp.get("passes") or []):
        lines += [
            "## Context Depth Profile",
            "- Note: These depth passes are workload-specific and not directly comparable as one ranking.",
        ]
        winners = dp.get("bucket_winners") or {}
        if winners:
            lines.append(
                "- Bucket winners: "
                f"`best_at_8k={winners.get('best_at_8k')}` · "
                f"`best_at_14k={winners.get('best_at_14k')}` · "
                f"`best_at_28k={winners.get('best_at_28k')}`"
            )
        for p in (dp.get("passes") or []):
            lines.append(
                f"- `{p.get('depth_label')}` prompt: winner=`{p.get('winner')}` "
                f"error=`{p.get('error')}` prompt_path=`{p.get('prompt_path')}`"
            )
        lines += [""]
    lines += [
        "## Product Boundary",
        "- OSS scope: fast config sweep + baseline deltas + lightweight smoke.",
        "- Use paid Sigilant Optimizer for full safety/quality certification, long-context reliability, and governance artifacts.",
        "",
    ]
    if repro:
        lines += ["## Reproduce", "```bash", str(repro), "```", ""]
    return "\n".join(lines)


def _build_frontier_svg(results: List[RunResult], context: Optional[Dict[str, Any]] = None) -> str:
    ok = [r for r in results if r.succeeded]
    if not ok:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='520'></svg>"

    # ── Canvas ────────────────────────────────────────────────────────────────
    w, h   = 1400, 760
    plot_x = 105
    plot_y = 72
    plot_w = 870
    plot_h = 550
    side_x = 1045
    side_y = 72
    side_w = 330
    side_h = 500

    # ── Data ─────────────────────────────────────────────────────────────────
    tps_vals  = [r.tps     for r in ok]
    ttft_vals = [r.ttft_ms for r in ok]
    ppl_vals  = [r.ppl     for r in ok if r.ppl is not None]
    min_tps, max_tps = min(tps_vals),  max(tps_vals)
    min_tt,  max_tt  = min(ttft_vals), max(ttft_vals)
    ppl_rng  = (min(ppl_vals), max(ppl_vals)) if ppl_vals else None

    # ── Nice round-number axis ranges ────────────────────────────────────────
    def _nice(lo: float, hi: float):
        span = max(hi - lo, 1e-6)
        raw  = span / 7.0
        mag  = 10.0 ** math.floor(math.log10(raw))
        step = mag
        for f in (1, 2, 5, 10, 20):
            step = mag * f
            if span / step <= 12:
                break
        return math.floor(lo / step) * step, math.ceil(hi / step) * step, step

    # 1.5% padding — just enough to keep points off the axis lines while keeping
    # nice_range close to the actual data so ticks land on clean round numbers.
    pad_tt  = max(max_tt  - min_tt,  1.0) * 0.015
    pad_tps = max(max_tps - min_tps, 1.0) * 0.015
    tt_lo,  tt_hi,  tt_step  = _nice(min_tt  - pad_tt,  max_tt  + pad_tt)
    tps_lo, tps_hi, tps_step = _nice(min_tps - pad_tps, max_tps + pad_tps)

    def sx(v: float) -> float: return plot_x + (v - tt_lo)  / (tt_hi  - tt_lo)  * plot_w
    def sy(v: float) -> float: return plot_y + plot_h - (v - tps_lo) / (tps_hi - tps_lo) * plot_h

    def bubble_r(ppl: Optional[float]) -> float:
        if ppl is None or ppl_rng is None or ppl_rng[1] == ppl_rng[0]: return 8.5
        t = (ppl - ppl_rng[0]) / (ppl_rng[1] - ppl_rng[0])
        # keep visual spread moderate (8.5–11.5) so bubbles remain readable
        return round(8.5 + (1.0 - t) * 3.0, 1)

    def short_q(q: str) -> str:
        s = (q or "").upper()
        if s == "FP16_BASELINE":
            return "FP16"
        if s == "INT8_W8A8":
            return "INT8"
        if s == "AWQ4_MARLIN":
            return "AWQ4"
        if s == "GPTQ4_MARLIN":
            return "GPTQ4"
        if s.endswith("_K_M"): return s[:-4]
        if s.endswith("_0"):   return s[:-2]
        return s

    def short_kv(kv: str) -> str:
        return "k16v16" if (kv or "").lower() == "k16v16" else "k8v8"

    QCOL = {
        "Q5_K_M": "#1E8E2E", "Q4_K_M": "#E59D0A", "Q3_K_M": "#E12121",
        "Q8_0": "#114FBE", "IQ3_M": "#188A3B",
        "FP16_BASELINE": "#1F7AFC", "INT8_W8A8": "#8C54FF",
        "AWQ4_MARLIN": "#F59E0B", "GPTQ4_MARLIN": "#10B981",
    }
    FB   = "#6B7280"

    ordered  = sorted(ok, key=lambda r: (r.ttft_ms, -r.tps))
    winner   = max(ok, key=lambda r: r.score or -1e9)
    fastest  = min(ok, key=lambda r: r.ttft_ms)
    rank_map = {id(r): i + 1 for i, r in enumerate(
        sorted(ok, key=lambda r: r.score or -1e9, reverse=True))}

    # Dynamic insight: compare fastest-TTFT config vs winner
    _fl = f"{short_q((fastest.config.quant_label or '').upper())} {fastest.config.context // 1024}k {short_kv(fastest.config.kv_type)}"
    _wl = f"{short_q((winner.config.quant_label or '').upper())}  {winner.config.context // 1024}k {short_kv(winner.config.kv_type)}"
    if id(fastest) == id(winner):
        _insight = [f"{_fl} wins", "fastest and best composite.", ""]
    elif fastest.ppl is not None and winner.ppl is not None and fastest.ppl > winner.ppl:
        _insight = [f"{_fl} is faster,", "but smaller bubble (higher PPL)", "lowers composite score."]
    else:
        _insight = [f"{_fl} has lowest TTFT,", f"but {_wl} wins on", "composite score."]

    # Legend quant families — only those present in this run
    _present = {(r.config.quant_label or "").upper() for r in ok}
    _qleg = [(q, c) for q, c in [
        ("Q5_K_M", "#1E8E2E"), ("Q4_K_M", "#E59D0A"), ("Q3_K_M", "#E12121"),
        ("Q8_0",   "#114FBE"), ("IQ3_M",  "#188A3B"),
        ("FP16_BASELINE", "#1F7AFC"), ("INT8_W8A8", "#8C54FF"),
        ("AWQ4_MARLIN", "#F59E0B"), ("GPTQ4_MARLIN", "#10B981"),
    ] if q in _present]

    rl = f" — Run {context['run_id']}" if context and context.get("run_id") else ""

    parts: List[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'>",
        "<rect width='100%' height='100%' fill='#ffffff'/>",
    ]

    # ── Title ─────────────────────────────────────────────────────────────────
    parts.append(
        f"<text x='{w // 2}' y='46' fill='#111' font-size='26' font-family='Arial' "
        f"font-weight='700' text-anchor='middle'>Sigilant Runner Frontier{rl}</text>"
    )

    # ── Grid + tick labels ────────────────────────────────────────────────────
    n = 0
    while True:
        t = tt_lo + n * tt_step
        if t > tt_hi + tt_step * 0.001: break
        gx = sx(t)
        if plot_x - 1 <= gx <= plot_x + plot_w + 1:
            parts.append(f"<line x1='{gx:.1f}' y1='{plot_y}' x2='{gx:.1f}' y2='{plot_y+plot_h}' stroke='#e4e8ef' stroke-width='1'/>")
            parts.append(f"<line x1='{gx:.1f}' y1='{plot_y+plot_h}' x2='{gx:.1f}' y2='{plot_y+plot_h+6}' stroke='#888'/>")
            parts.append(f"<text x='{gx:.0f}' y='{plot_y+plot_h+22}' fill='#555' font-size='13' font-family='Arial' text-anchor='middle'>{int(round(t))}</text>")
        n += 1

    n = 0
    while True:
        t = tps_lo + n * tps_step
        if t > tps_hi + tps_step * 0.001: break
        gy = sy(t)
        if plot_y - 1 <= gy <= plot_y + plot_h + 1:
            parts.append(f"<line x1='{plot_x}' y1='{gy:.1f}' x2='{plot_x+plot_w}' y2='{gy:.1f}' stroke='#e4e8ef' stroke-width='1'/>")
            parts.append(f"<line x1='{plot_x-6}' y1='{gy:.1f}' x2='{plot_x}' y2='{gy:.1f}' stroke='#888'/>")
            parts.append(f"<text x='{plot_x-10}' y='{gy+4:.1f}' fill='#555' font-size='13' font-family='Arial' text-anchor='end'>{t:.0f}</text>")
        n += 1

    # ── Axis borders ──────────────────────────────────────────────────────────
    parts.append(f"<line x1='{plot_x}' y1='{plot_y}' x2='{plot_x}' y2='{plot_y+plot_h}' stroke='#666' stroke-width='1.5'/>")
    parts.append(f"<line x1='{plot_x}' y1='{plot_y+plot_h}' x2='{plot_x+plot_w}' y2='{plot_y+plot_h}' stroke='#666' stroke-width='1.5'/>")

    # ── Axis labels ───────────────────────────────────────────────────────────
    parts.append(f"<text x='{plot_x+plot_w//2}' y='{plot_y+plot_h+46}' fill='#333' font-size='15' font-family='Arial' text-anchor='middle'>TTFT (ms, lower is better)</text>")
    rcx, rcy = plot_x - 70, plot_y + plot_h // 2
    parts.append(f"<text x='{rcx}' y='{rcy}' fill='#333' font-size='15' font-family='Arial' transform='rotate(-90 {rcx} {rcy})' text-anchor='middle'>TPS (higher is better)</text>")

    # ── Dashed frontier trail ─────────────────────────────────────────────────
    for a, b in zip(ordered, ordered[1:]):
        parts.append(
            f"<line x1='{sx(a.ttft_ms):.1f}' y1='{sy(a.tps):.1f}' "
            f"x2='{sx(b.ttft_ms):.1f}' y2='{sy(b.tps):.1f}' "
            "stroke='#b0b8c4' stroke-width='1.8' stroke-dasharray='7,5'/>")

    # ── Data points ───────────────────────────────────────────────────────────
    for r in ordered:
        x, y = sx(r.ttft_ms), sy(r.tps)
        c    = QCOL.get((r.config.quant_label or "").upper(), FB)
        kv   = (r.config.kv_type or "").lower()
        rr   = bubble_r(r.ppl)
        if kv == "k8v8":
            parts.append(
                f"<polygon points='{x:.1f},{y-rr:.1f} {x+rr:.1f},{y:.1f} "
                f"{x:.1f},{y+rr:.1f} {x-rr:.1f},{y:.1f}' fill='{c}' stroke='white' stroke-width='1.5'/>")
        else:
            parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{rr:.1f}' fill='{c}' stroke='white' stroke-width='1.5'/>")

    # ── Winner ring + green star ───────────────────────────────────────────────
    wx, wy = sx(winner.ttft_ms), sy(winner.tps)
    parts.append(f"<circle cx='{wx:.1f}' cy='{wy:.1f}' r='20' fill='none' stroke='#111' stroke-width='4.5'/>")
    parts.append(
        f"<text x='{wx:.1f}' y='{wy+6:.1f}' fill='#1E8E2E' stroke='white' stroke-width='0.5' "
        "font-size='15' text-anchor='middle' font-family='Arial'>★</text>")

    # ── Label placement: greedy collision avoidance ────────────────────────────
    CW, LH = 7.0, 15

    def _lbox(lx: float, ly: float, text: str, anc: str):
        tw = len(text) * CW
        if anc == "end":    return (lx - tw, ly - LH, lx,        ly + 4)
        if anc == "middle": return (lx - tw/2, ly - LH, lx + tw/2, ly + 4)
        return                     (lx,        ly - LH, lx + tw,   ly + 4)

    def _ov(b1, b2, pad: float = 3.0) -> bool:
        return b1[0]-pad < b2[2] and b1[2]+pad > b2[0] and b1[1]-pad < b2[3] and b1[3]+pad > b2[1]

    # Safe region: strictly inside the axis box
    _lx_min = plot_x + 4
    _lx_max = plot_x + plot_w - 4
    _ly_min = plot_y + LH + 2
    _ly_max = plot_y + plot_h - 4

    # Cardinal-first angles: R L U D then diagonals, then fine-grained
    ANGLES = [0, 180, 90, 270, 45, 135, 315, 225, 30, 150, 60, 120, 210, 330, 240, 300]
    DISTS  = [48, 62, 78, 96, 116, 140]

    # Seed obstacles with all data-point bounding boxes
    placed = [
        (sx(r.ttft_ms) - bubble_r(r.ppl) - 5, sy(r.tps) - bubble_r(r.ppl) - 5,
         sx(r.ttft_ms) + bubble_r(r.ppl) + 5, sy(r.tps) + bubble_r(r.ppl) + 5)
        for r in ordered
    ]

    # Process highest-ranked (winner area) first so they get best spots
    order_by_rank = sorted(ordered, key=lambda r: rank_map[id(r)])

    lspecs_map = {}
    for r in order_by_rank:
        px, py = sx(r.ttft_ms), sy(r.tps)
        rank   = rank_map[id(r)]
        q      = (r.config.quant_label or "").upper()
        ctxk   = int(r.config.context // 1024)
        kvs    = short_kv(r.config.kv_type)
        pt     = f"{r.ppl:.2f}" if r.ppl is not None else "n/a"
        lbl    = f"{short_q(q)} {ctxk}k {kvs} ({pt})"
        col    = QCOL.get(q, "#1f2937")

        blx = bly = banc = bboxv = None
        bsc = 1e18

        # Prefer label placement to the side with more free space.
        pref_right = px < (plot_x + plot_w * 0.58)
        angs = ANGLES if pref_right else [180, 150, 210, 135, 225, 120, 240, 90, 270, 60, 300, 30, 330, 0, 45, 315]
        for dist in DISTS:
            for adeg in angs:
                a   = math.radians(adeg)
                lx_ = px + dist * math.cos(a)
                ly_ = py - dist * math.sin(a)  # SVG y-inverted: -sin gives "up"
                ca  = math.cos(a)
                anc = "start" if ca > 0.25 else ("end" if ca < -0.25 else "middle")
                bb  = _lbox(lx_, ly_, lbl, anc)
                # strict: every corner inside axis box
                if bb[0] < _lx_min or bb[2] > _lx_max or bb[1] < _ly_min or bb[3] > _ly_max:
                    continue
                nov = sum(1 for b in placed if _ov(bb, b))
                sc  = nov * 1e6 + dist
                if sc < bsc:
                    bsc = sc; blx = lx_; bly = ly_; banc = anc; bboxv = bb
            if bsc < 1e6:
                break

        if blx is None:  # fallback: best we found even if it overlaps
            blx, bly, banc = px + 50, py - 18, "start"
            bboxv = _lbox(blx, bly, lbl, banc)

        placed.append(bboxv)
        lspecs_map[id(r)] = (blx, bly, banc, lbl, col)

    lspecs = [(r,) + lspecs_map[id(r)] for r in ordered]

    for r, lx_, ly_, anc, lbl, col in lspecs:
        px, py = sx(r.ttft_ms), sy(r.tps)
        parts.append(f"<line x1='{px:.1f}' y1='{py:.1f}' x2='{lx_:.1f}' y2='{ly_:.1f}' stroke='{col}' stroke-width='1' opacity='0.65'/>")
        parts.append(f"<text x='{lx_:.1f}' y='{ly_:.1f}' fill='{col}' font-size='13' font-family='Arial' text-anchor='{anc}'>{lbl}</text>")

    # ── Right side: 5 stacked boxes aligned to chart height ──────────────────
    gap = 12
    stack_top = plot_y
    stack_bottom = plot_y + plot_h
    stack_h = stack_bottom - stack_top
    box_w = side_w
    box_x = side_x

    # custom vertical allocation (sum = 1.0) for cleaner balance
    # 1: color, 2: shape, 3: bubble, 4: winner, 5: insight
    ratios = [0.19, 0.17, 0.25, 0.17, 0.22]
    usable_h = stack_h - 4 * gap
    heights = [usable_h * r for r in ratios]

    def _box_y(i: int) -> float:
        y0 = stack_top
        for k in range(i):
            y0 += heights[k] + gap
        return y0

    # Box style
    box_bg = "#fdfdfd"
    box_stroke = "#c9cfd8"
    title_color = "#111"
    body_color = "#222"

    # 1) COLOR box
    y = _box_y(0)
    box_h = heights[0]
    parts.append(f"<rect x='{box_x}' y='{y:.1f}' width='{box_w}' height='{box_h:.1f}' rx='8' ry='8' fill='{box_bg}' stroke='{box_stroke}' stroke-width='1.5'/>")
    cx = box_x + box_w / 2.0
    parts.append(f"<text x='{cx:.1f}' y='{y+24:.1f}' fill='{title_color}' font-size='13' font-family='Arial' font-weight='700' text-anchor='middle'>COLOR = QUANT FAMILY</text>")
    ly2 = y + 42
    # 2 columns to avoid overlap
    col1_x = box_x + 30
    col2_x = box_x + box_w / 2.0 + 8
    row_gap = 20
    qleg = _qleg[:4]
    if len(qleg) < 4:
        qleg = qleg + [("", "#000")] * (4 - len(qleg))
    for i, (q2, c2) in enumerate(qleg[:2]):
        if q2:
            yy = ly2 + i * row_gap
            parts.append(f"<circle cx='{col1_x}' cy='{yy-4:.1f}' r='6' fill='{c2}'/>")
            parts.append(f"<text x='{col1_x+14}' y='{yy:.1f}' fill='{body_color}' font-size='11' font-family='Arial'>{short_q(q2)}</text>")
    for i, (q2, c2) in enumerate(qleg[2:4]):
        if q2:
            yy = ly2 + i * row_gap
            parts.append(f"<circle cx='{col2_x:.1f}' cy='{yy-4:.1f}' r='6' fill='{c2}'/>")
            parts.append(f"<text x='{col2_x+14:.1f}' y='{yy:.1f}' fill='{body_color}' font-size='11' font-family='Arial'>{short_q(q2)}</text>")

    # 2) SHAPE box
    y = _box_y(1)
    box_h = heights[1]
    parts.append(f"<rect x='{box_x}' y='{y:.1f}' width='{box_w}' height='{box_h:.1f}' rx='8' ry='8' fill='{box_bg}' stroke='{box_stroke}' stroke-width='1.5'/>")
    lx2, ly2 = box_x + 18, y + 24
    cx = box_x + box_w / 2.0
    parts.append(f"<text x='{cx:.1f}' y='{ly2:.1f}' fill='{title_color}' font-size='13' font-family='Arial' font-weight='700' text-anchor='middle'>SHAPE = KV TYPE</text>")
    ly2 += 26
    parts.append(f"<circle cx='{lx2+20}' cy='{ly2-5:.1f}' r='7' fill='#555'/>")
    parts.append(f"<text x='{lx2+36}' y='{ly2:.1f}' fill='{body_color}' font-size='11' font-family='Arial'>circle = k16v16 (higher quality)</text>")
    ly2 += 24
    parts.append(f"<polygon points='{lx2+20},{ly2-12:.1f} {lx2+27},{ly2-5:.1f} {lx2+20},{ly2+2:.1f} {lx2+13},{ly2-5:.1f}' fill='#555'/>")
    parts.append(f"<text x='{lx2+36}' y='{ly2:.1f}' fill='{body_color}' font-size='11' font-family='Arial'>diamond = k8v8 (lower memory)</text>")

    # 3) BUBBLE box
    y = _box_y(2)
    box_h = heights[2]
    parts.append(f"<rect x='{box_x}' y='{y:.1f}' width='{box_w}' height='{box_h:.1f}' rx='8' ry='8' fill='{box_bg}' stroke='{box_stroke}' stroke-width='1.5'/>")
    lx2, ly2 = box_x + 18, y + 24
    cx = box_x + box_w / 2.0
    parts.append(f"<text x='{cx:.1f}' y='{ly2:.1f}' fill='{title_color}' font-size='13' font-family='Arial' font-weight='700' text-anchor='middle'>BUBBLE SIZE = INVERSE PPL</text>")
    ly2 += 22
    parts.append(f"<text x='{cx:.1f}' y='{ly2:.1f}' fill='#555' font-size='11' font-family='Arial' text-anchor='middle'>Bigger bubble = better PPL (lower)</text>")
    ly2 += 28
    for i2, rr2 in enumerate([3, 5, 7, 9, 11]):
        cx2 = lx2 + 8 + i2 * 46
        parts.append(f"<circle cx='{cx2}' cy='{ly2:.1f}' r='{rr2}' fill='#aaa'/>")
    ly2 += 16
    parts.append(f"<line x1='{lx2+2}' y1='{ly2:.1f}' x2='{lx2+240}' y2='{ly2:.1f}' stroke='#444'/>")
    parts.append(f"<polygon points='{lx2+240},{ly2:.1f} {lx2+234},{ly2-4:.1f} {lx2+234},{ly2+4:.1f}' fill='#444'/>")
    ly2 += 18
    parts.append(f"<text x='{lx2}' y='{ly2:.1f}' fill='#555' font-size='11' font-family='Arial'>Worse PPL</text>")
    parts.append(f"<text x='{lx2+165}' y='{ly2:.1f}' fill='#555' font-size='11' font-family='Arial'>Better PPL</text>")

    # 4) WINNER box
    y = _box_y(3)
    box_h = heights[3]
    parts.append(f"<rect x='{box_x}' y='{y:.1f}' width='{box_w}' height='{box_h:.1f}' rx='8' ry='8' fill='{box_bg}' stroke='{box_stroke}' stroke-width='1.5'/>")
    lx2, ly2 = box_x + 18, y + 24
    cx = box_x + box_w / 2.0
    parts.append(f"<text x='{cx:.1f}' y='{ly2:.1f}' fill='{title_color}' font-size='13' font-family='Arial' font-weight='700' text-anchor='middle'>WINNER</text>")
    ly2 += 32
    parts.append(f"<circle cx='{lx2+15}' cy='{ly2-12:.1f}' r='13' fill='white' stroke='black' stroke-width='3.5'/>")
    parts.append(f"<text x='{lx2+15}' y='{ly2-6:.1f}' fill='#1E8E2E' font-size='13' text-anchor='middle' font-family='Arial'>★</text>")
    parts.append(f"<text x='{lx2+36}' y='{ly2-12:.1f}' fill='{body_color}' font-size='12' font-family='Arial'>Black ring + star</text>")
    parts.append(f"<text x='{lx2+36}' y='{ly2+4:.1f}' fill='{body_color}' font-size='12' font-family='Arial'>= Best overall score</text>")

    # 5) INSIGHT box
    y = _box_y(4)
    box_h = heights[4]
    parts.append(f"<rect x='{box_x}' y='{y:.1f}' width='{box_w}' height='{box_h:.1f}' rx='8' ry='8' fill='#fffaf0' stroke='#f3b65a' stroke-width='1.5'/>")
    lx2 = box_x + 12
    text_y1 = y + box_h * 0.38
    text_y2 = y + box_h * 0.62
    bulb_y = (text_y1 + text_y2) / 2.0 - 6
    parts.append(f"<text x='{lx2}' y='{bulb_y:.1f}' fill='#8a6405' font-size='18' font-family='Arial'>💡</text>")
    parts.append(f"<text x='{lx2+26}' y='{text_y1:.1f}' fill='{body_color}' font-size='12' font-family='Arial'>{_insight[0]}</text>")
    parts.append(f"<text x='{lx2+26}' y='{text_y2:.1f}' fill='{body_color}' font-size='12' font-family='Arial'>{_insight[1]}</text>")
    if _insight[2]:
        parts.append(f"<text x='{lx2+26}' y='{y + box_h * 0.82:.1f}' fill='{body_color}' font-size='12' font-family='Arial'>{_insight[2]}</text>")

    # ── Footer formula ────────────────────────────────────────────────────────
    # nudged slightly lower for better separation from x-axis label
    fy = plot_y + plot_h + 72
    parts.append(f"<text x='80' y='{fy}' fill='#6b7280' font-size='12' font-style='italic' font-family='Arial'>Score formula: TTFT_norm = 0.5×(min TTFT p50 / TTFT p50) + 0.5×(min TTFT p95 / TTFT p95);</text>")
    parts.append(f"<text x='80' y='{fy+16}' fill='#6b7280' font-size='12' font-style='italic' font-family='Arial'>Final score = 40% TPS_norm + 20% TTFT_norm + 40% PPL_norm (or 60/40 when PPL unavailable).</text>")

    parts.append("</svg>")
    return "\n".join(parts)


def build_repro_command(*, model: str, backend: str, engine: str, hardware: str, params_b: Optional[float], max_configs: int, trials: int, score_profile: str = "balanced") -> str:
    cmd = [
        "sigilant-runner", "run",
        "--model", model,
        "--backend", backend,
        "--engine", engine,
        "--hardware", hardware,
        "--configs", str(max_configs),
        "--trials", str(trials),
        "--score-profile", str(score_profile),
    ]
    if params_b is not None:
        cmd += ["--params-b", str(params_b)]
    return " ".join(shlex.quote(x) for x in cmd)


def _build_terminal_like(payload: Dict[str, Any]) -> str:
    ctx = payload.get("context") or {}
    rows = payload.get("results") or []
    ok_sorted = sorted(
        [r for r in rows if not r.get("error")],
        key=lambda x: (
            -float(x.get("score") or 0.0),
            float(x.get("ttft_ms") or 1e12),
            -float(x.get("tps") or 0.0),
            float(x.get("ppl") if x.get("ppl") is not None else 1e9),
            str(x.get("config", {}).get("label") or ""),
        ),
    )
    smoke = payload.get("agent_smoke") or {}
    bc = payload.get("baseline_compare") or {}
    ci = payload.get("confidence_inputs") or {}
    dp = payload.get("depth_profile") or {}
    lines = []
    lines.append(
        f"sigilant-runner · {ctx.get('model_label') or ctx.get('model')} · "
        f"{ctx.get('hardware')} · {ctx.get('engine')} · {len(ok_sorted)} configs"
    )
    lines.append("")
    lines.append("Config | TPS | TPS p95 | TTFT | TTFT p95 | ITL | PPL | TPS% | TTFT% | PPL% | Score")
    for i, r in enumerate(ok_sorted):
        label = r["config"]["label"] + ("  <- best" if i == 0 else "")
        lines.append(
            f"{label} | {_fmt_num(r.get('tps'), 1)} | {_fmt_num(r.get('tps_p95'), 1)} | {_fmt_num(r.get('ttft_ms'), 1)} | "
            f"{_fmt_num(r.get('ttft_p95_ms'), 1)} | {_fmt_num(r.get('itl_ms'), 2)} | {_fmt_num(r.get('ppl'), 2)} | "
            f"{_fmt_num(r.get('tps_norm'), 1)} | {_fmt_num(r.get('ttft_norm'), 1)} | {_fmt_num(r.get('ppl_norm'), 1)} | {r.get('score')}"
        )
    if ok_sorted:
        lines.append("")
        lines.append(f"Best config: {ok_sorted[0]['config']['label']}")
    if bc.get("found"):
        lines.append(
            "Auto baseline compare: "
            f"score Δ={_fmt_num(bc.get('delta_score'), 2)} TPS Δ={_fmt_num(bc.get('delta_tps'), 2)} "
            f"TTFT Δ={_fmt_num(bc.get('delta_ttft_ms'), 1)}ms PPL Δ={_fmt_num(bc.get('delta_ppl'), 2)}"
        )
        if bc.get("delta_tps_p95") is not None or bc.get("delta_ttft_p95_ms") is not None:
            lines.append(
                f"                     TPS p95 Δ={_fmt_num(bc.get('delta_tps_p95'), 2)} "
                f"TTFT p95 Δ={_fmt_num(bc.get('delta_ttft_p95_ms'), 1)}ms"
            )
    if isinstance(smoke, dict) and smoke:
        lines.append(
            f"Agent smoke: {smoke.get('passed')}/{smoke.get('total')} ({_fmt_pct(smoke.get('pass_rate'))}) "
            f"[{smoke.get('diagnosis')}]"
        )
    if isinstance(ci, dict) and ci:
        lines.append(
            "Confidence: "
            f"target={ci.get('confidence_target')} "
            f"gap_before={_fmt_num(ci.get('gap_pct_before'), 2)}% "
            f"var_before={_fmt_num(ci.get('variance_pct_before'), 2)}% "
            f"replay={ci.get('replay_triggered')}({ci.get('replay_outcome')}) "
            f"gap_after={_fmt_num(ci.get('gap_pct_after'), 2)}%"
        )
    if isinstance(dp, dict) and (dp.get("passes") or []):
        lines.append("")
        lines.append("Depth profile (not cross-depth comparable):")
        winners = dp.get("bucket_winners") or {}
        if winners:
            lines.append(
                "  - bucket winners: "
                f"best_at_8k={winners.get('best_at_8k') or 'n/a'} | "
                f"best_at_14k={winners.get('best_at_14k') or 'n/a'} | "
                f"best_at_28k={winners.get('best_at_28k') or 'n/a'}"
            )
        for p in (dp.get("passes") or []):
            lines.append(
                f"  - {p.get('depth_label')}: winner={p.get('winner') or 'n/a'} "
                f"error={p.get('error') or 'none'}"
            )
    lines.append("")
    lines.append("PPL is a quality proxy, not production validation.")
    lines.append("Full production safety and long-context certification require Sigilant Optimizer.")
    return "\n".join(lines) + "\n"
