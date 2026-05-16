from __future__ import annotations

from typing import List

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.metrics import RunResult
from ..core.ranking import ranked_succeeded

console = Console()


def print_header(model: str, hardware: str, engine: str, n_configs: int) -> None:
    console.print()
    console.print(
        f"[bold cyan]sigilant-runner[/bold cyan]  "
        f"[white]{model}[/white]  ·  "
        f"[dim]{hardware}[/dim]  ·  "
        f"[dim]{engine}[/dim]  ·  "
        f"[dim]{n_configs} configs[/dim]"
    )
    console.print()


def print_results_table(results: List[RunResult], *, profile: str = "balanced", w_tps: float = 0.40, w_ttft: float = 0.20, w_ppl: float = 0.40) -> None:
    succeeded = ranked_succeeded(results)
    failed = [r for r in results if not r.succeeded]
    has_tps_p95 = any((r.tps_p95 is not None) for r in succeeded)
    has_ttft_p95 = any((r.ttft_p95_ms is not None) for r in succeeded)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("Config",   min_width=42)
    table.add_column("TPS p50" if has_tps_p95 else "TPS", justify="right", style="cyan")
    if has_tps_p95:
        table.add_column("TPS p95", justify="right", style="cyan")
    table.add_column("TTFT p50" if has_ttft_p95 else "TTFT", justify="right", style="cyan")
    if has_ttft_p95:
        table.add_column("TTFT p95", justify="right", style="cyan")
    table.add_column("ITL",      justify="right", style="dim")
    table.add_column("PPL",      justify="right", style="cyan")
    table.add_column("TPS%",     justify="right", style="dim")
    table.add_column("TTFT%",    justify="right", style="dim")
    table.add_column("PPL%",     justify="right", style="dim")
    table.add_column("Score",    justify="right", style="bold")

    for i, r in enumerate(succeeded):
        ppl_str   = f"{r.ppl:.2f}" if r.ppl is not None else "—"
        tpsn_str  = f"{r.tps_norm:.1f}" if r.tps_norm is not None else "—"
        ttfn_str  = f"{r.ttft_norm:.1f}" if r.ttft_norm is not None else "—"
        ppln_str  = f"{r.ppl_norm:.1f}" if r.ppl_norm is not None else "—"
        score_str = str(r.score)   if r.score is not None else "—"
        itl_str   = f"{r.itl_ms:.1f}ms"
        tps_p95_str = f"{r.tps_p95:.1f}" if r.tps_p95 is not None else "—"
        ttft_p95_str = f"{r.ttft_p95_ms:.0f}ms" if r.ttft_p95_ms is not None else "—"

        if i == 0:
            label = Text(r.config.label() + "  ← best", style="bold green")
            score = Text(score_str, style="bold green")
        else:
            label = Text(r.config.label())
            score = Text(score_str)

        row = [label, f"{r.tps:.1f}"]
        if has_tps_p95:
            row.append(tps_p95_str)
        row.append(f"{r.ttft_ms:.0f}ms")
        if has_ttft_p95:
            row.append(ttft_p95_str)
        row.extend([itl_str, ppl_str, tpsn_str, ttfn_str, ppln_str, score])
        table.add_row(*row)

    if failed:
        table.add_section()
        for r in failed:
            row = [Text(r.config.label(), style="dim"), "—"]
            if has_tps_p95:
                row.append("—")
            row.append("—")
            if has_ttft_p95:
                row.append("—")
            row.extend(["—", "—", "—", "—", "—", Text("FAILED", style="red")])
            table.add_row(*row)

    console.print(table)

    if succeeded:
        best = succeeded[0]
        console.print(f"[bold]Best config:[/bold]  {best.config.label()}")
    if has_ttft_p95 and has_tps_p95:
        console.print(
            "[dim]Score formula: TPS_norm = 0.5×(TPS p50 / max TPS p50) + 0.5×(TPS p95 / max TPS p95); "
            "TTFT_norm = 0.5×(min TTFT p50 / TTFT p50) + 0.5×(min TTFT p95 / TTFT p95); "
            f"final score ({profile}) = {int(w_tps*100)}% TPS_norm + {int(w_ttft*100)}% TTFT_norm + {int(w_ppl*100)}% PPL_norm "
            "(PPL unavailable -> TPS/TTFT renormalized).[/dim]"
        )
    elif has_ttft_p95:
        console.print(
            "[dim]Score formula: TPS_norm = TPS p50 / max TPS p50; "
            "TTFT_norm = 0.5×(min TTFT p50 / TTFT p50) + 0.5×(min TTFT p95 / TTFT p95); "
            f"final score ({profile}) = {int(w_tps*100)}% TPS_norm + {int(w_ttft*100)}% TTFT_norm + {int(w_ppl*100)}% PPL_norm "
            "(PPL unavailable -> TPS/TTFT renormalized).[/dim]"
        )
    else:
        console.print(
            f"[dim]Score formula ({profile}): {int(w_tps*100)}% TPS_norm + {int(w_ttft*100)}% TTFT_norm + "
            f"{int(w_ppl*100)}% PPL_norm (PPL unavailable -> TPS/TTFT renormalized).[/dim]"
        )

    console.print()
    console.print("[dim]PPL is a quality proxy, not production validation.[/dim]")
    console.print()
    console.print("[yellow bold]! Agent safety NOT fully evaluated.[/yellow bold]")
    console.print("[dim]  Agent smoke helps triage model-limited vs harness-limited issues.[/dim]")
    console.print("[dim]  Full production safety and long-context certification require Sigilant Optimizer.[/dim]")
    console.print()
    console.print("[dim]  → sigilantlabs.com/optimize[/dim]")
    console.print()
