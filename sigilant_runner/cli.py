from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from . import __version__

app = typer.Typer(
    name="sigilant-runner",
    help="Open-source LLM inference sweep — TPS, TTFT, ITL, PPL across 16 configs.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

console = Console()


@app.command()
def run(
    model: str = typer.Option(
        ..., "--model", "-m",
        help="HuggingFace repo ID (e.g. mistralai/Mistral-7B-Instruct-v0.3) or path to a local .gguf file.",
    ),
    backend: str = typer.Option(
        "local", "--backend", "-b",
        help="Where to run: local | modal",
    ),
    engine: str = typer.Option(
        "llama.cpp", "--engine", "-e",
        help="Inference engine: llama.cpp",
    ),
    hardware: str = typer.Option(
        "auto", "--hardware",
        help=(
            "Target GPU. 'auto' detects local hardware. "
            "Cloud values: a10g, a100, h100, rtx4090, rtx3090, rtxa6000, t4, l4"
        ),
    ),
    params_b: Optional[float] = typer.Option(
        None, "--params-b",
        help="Model size in billions of parameters (auto-detected from model name if omitted).",
    ),
    max_configs: int = typer.Option(
        16, "--configs",
        help="Maximum number of configs to sweep.",
    ),
    trials: int = typer.Option(
        12, "--trials",
        help="Number of timed trials per config. Default is 12 for stable winner ranking.",
    ),
    confidence_target: str = typer.Option(
        "medium", "--confidence-target",
        help="Confidence label preference (reported in artifacts).",
    ),
    score_profile: str = typer.Option(
        "balanced", "--score-profile",
        help="Scoring preset: balanced | latency | quality.",
    ),
    benchmark_mode: str = typer.Option(
        "ranking", "--benchmark-mode",
        help="Benchmark mode: ranking | depth_profile.",
    ),
    depth_prompt_8k: str = typer.Option(
        "prompts/hard_quality_8k_prompt.txt", "--depth-prompt-8k",
        help="Prompt file for 8k depth pass (used in --benchmark-mode depth_profile).",
    ),
    depth_prompt_14k: str = typer.Option(
        "prompts/hard_quality_14k_prompt.txt", "--depth-prompt-14k",
        help="Prompt file for 14k depth pass (used in --benchmark-mode depth_profile).",
    ),
    depth_prompt_28k: str = typer.Option(
        "prompts/hard_quality_28k_prompt.txt", "--depth-prompt-28k",
        help="Prompt file for 28k depth pass (used in --benchmark-mode depth_profile).",
    ),
    baseline_config: Optional[str] = typer.Option(
        None, "--baseline-config",
        help="Compare recommended winner vs current config. Format: QUANT,CTX,KV,REGIME (e.g. Q4_K_M,8192,k16v16,default)",
    ),
    check: bool = typer.Option(
        False, "--check",
        help="CI mode: fail non-zero if winner drifts from baseline file.",
    ),
    baseline_file: str = typer.Option(
        ".sigilant_baseline.json", "--baseline-file",
        help="Path to baseline winner file used by --check / --update-baseline.",
    ),
    update_baseline: bool = typer.Option(
        False, "--update-baseline",
        help="Write current winner to baseline file.",
    ),
    agent_smoke: bool = typer.Option(
        False, "--agent-smoke",
        help="Run lightweight agent smoke checks on the winner (llama.cpp on local/modal).",
    ),
    json_out: bool = typer.Option(
        False, "--json",
        help="Write results to sigilant_results.json in addition to terminal output.",
    ),
) -> None:
    """Run a 16-config inference sweep and rank by Sigilant Score."""
    from .core.hardware import detect_hardware, KNOWN_VRAM
    from .core.grid import generate_grid
    from .core.scoring import compute_scores, resolve_weight_profile
    from .models import resolve, infer_params_b
    from .output.table import print_header, print_results_table
    from .output.export import export_json, export_bundle, build_repro_command
    from .core.diagnostics import build_stability_report
    from .core.ranking import ranked_succeeded

    # ── Hardware ──────────────────────────────────────────────────────────────
    if hardware == "auto":
        hw      = detect_hardware()
        vram_gb = hw.vram_gb
        hw_label = f"{hw.gpu_name} {hw.vram_gb}GB" if hw.vram_gb > 0 else hw.gpu_name
    else:
        vram_gb  = KNOWN_VRAM.get(hardware.lower(), 24.0)
        hw_label = hardware.upper()

    if backend == "local" and vram_gb == 0:
        console.print(
            "[yellow]No GPU detected — running on CPU. "
            "Inference will be slow. Consider --backend modal.[/yellow]\n"
        )
        vram_gb = 64.0  # Allow large context on CPU (RAM-backed)

    # ── Model resolution ──────────────────────────────────────────────────────
    effective_params_b = params_b if params_b is not None else infer_params_b(model)

    console.print("[dim]Resolving model...[/dim]")
    try:
        models, model_label, repo_id = resolve(model, engine, backend=backend)
    except Exception as exc:
        console.print(f"[red]Model resolution failed:[/red] {exc}")
        raise typer.Exit(1)

    # ── Grid ──────────────────────────────────────────────────────────────────
    grid = generate_grid(
        models=models,
        vram_gb=vram_gb,
        model_params_b=effective_params_b,
        max_configs=max_configs,
        model_repo=repo_id,
        engine=engine,
    )

    if not grid:
        console.print(
            "[red]No configs fit within available VRAM.[/red] "
            "Try a smaller model, a higher-VRAM GPU, or --params-b with the correct size."
        )
        raise typer.Exit(1)

    print_header(model_label, hw_label, engine, len(grid))

    # ── Run ───────────────────────────────────────────────────────────────────
    resolved_trials = max(1, int(trials))

    profile = (score_profile or "balanced").strip().lower()
    if profile not in {"balanced", "latency", "quality"}:
        console.print("[red]Invalid --score-profile.[/red] Use: balanced | latency | quality")
        raise typer.Exit(1)
    mode = (benchmark_mode or "ranking").strip().lower()
    if mode not in {"ranking", "depth_profile"}:
        console.print("[red]Invalid --benchmark-mode.[/red] Use: ranking | depth_profile")
        raise typer.Exit(1)
    if engine != "llama.cpp":
        console.print("[red]Invalid --engine for this build.[/red] Use: llama.cpp")
        raise typer.Exit(1)
    if backend not in {"local", "modal"}:
        console.print("[red]Invalid --backend for this build.[/red] Use: local | modal")
        raise typer.Exit(1)

    # Ranking pass (single fixed workload)
    # In depth_profile mode, force ranking pass to 8k prompt and treat it as the
    # 8k bucket winner; cross-depth global winner is intentionally avoided.
    prev_prompt_env = os.environ.get("SIGILANT_BENCH_PROMPT_FILE")
    try:
        if mode == "depth_profile":
            p8 = Path(depth_prompt_8k)
            if p8.exists():
                os.environ["SIGILANT_BENCH_PROMPT_FILE"] = str(p8.resolve())
        results = _dispatch(backend, engine, grid, trials=resolved_trials)
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)
    finally:
        if prev_prompt_env is None:
            os.environ.pop("SIGILANT_BENCH_PROMPT_FILE", None)
        else:
            os.environ["SIGILANT_BENCH_PROMPT_FILE"] = prev_prompt_env

    # Score ranking-pass results immediately so depth bucket winner(8k) can be derived.
    results = compute_scores(results, profile=profile)

    depth_profile_payload = None
    if mode == "depth_profile":
        # 8k was already executed as the ranking pass above; do not rerun it.
        depth_specs = [
            ("14k", depth_prompt_14k),
            ("28k", depth_prompt_28k),
        ]
        depth_runs: List[Dict[str, Any]] = []
        prev_prompt_env = os.environ.get("SIGILANT_BENCH_PROMPT_FILE")
        try:
            # Include 8k pass from already-scored ranking results without extra GPU run.
            dw = _best_result(results)
            depth_runs.append({
                "depth_label": "8k",
                "prompt_path": str(Path(depth_prompt_8k).resolve()) if Path(depth_prompt_8k).exists() else depth_prompt_8k,
                "error": None,
                "winner": (dw.config.label() if dw else None),
                "results": [
                    {
                        "label": r.config.label(),
                        "score": r.score,
                        "tps": r.tps,
                        "ttft_ms": r.ttft_ms,
                        "ppl": r.ppl,
                        "status": "pass" if r.succeeded else "failed",
                        "error": r.error,
                    }
                    for r in results
                ],
                "derived_from_ranking": True,
            })
            for dlabel, dpath in depth_specs:
                p = Path(dpath)
                if not p.exists():
                    depth_runs.append({
                        "depth_label": dlabel,
                        "prompt_path": dpath,
                        "error": "prompt_missing",
                        "winner": None,
                        "results": [],
                    })
                    continue
                os.environ["SIGILANT_BENCH_PROMPT_FILE"] = str(p.resolve())
                try:
                    dresults = _dispatch(backend, engine, grid, trials=resolved_trials)
                    dresults = compute_scores(dresults, profile=profile)
                    dw = _best_result(dresults)
                    depth_runs.append({
                        "depth_label": dlabel,
                        "prompt_path": str(p.resolve()),
                        "error": None,
                        "winner": (dw.config.label() if dw else None),
                        "results": [
                            {
                                "label": r.config.label(),
                                "score": r.score,
                                "tps": r.tps,
                                "ttft_ms": r.ttft_ms,
                                "ppl": r.ppl,
                                "status": "pass" if r.succeeded else "failed",
                                "error": r.error,
                            }
                            for r in dresults
                        ],
                    })
                except Exception as exc:
                    depth_runs.append({
                        "depth_label": dlabel,
                        "prompt_path": str(p.resolve()),
                        "error": f"{type(exc).__name__}: {exc}",
                        "winner": None,
                        "results": [],
                    })
        finally:
            if prev_prompt_env is None:
                os.environ.pop("SIGILANT_BENCH_PROMPT_FILE", None)
            else:
                os.environ["SIGILANT_BENCH_PROMPT_FILE"] = prev_prompt_env
        best_by_bucket = {f"best_at_{p.get('depth_label')}": p.get("winner") for p in depth_runs}
        depth_profile_payload = {
            "mode": "depth_profile",
            "note": "Depth passes are workload-specific and not directly comparable with each other.",
            "bucket_winners": best_by_bucket,
            "passes": depth_runs,
        }
    w_tps, w_ttft, w_ppl = resolve_weight_profile(profile)

    stability = build_stability_report(results)
    confidence_inputs = _build_confidence_inputs(
        results=results,
        stability=stability,
        confidence_target=confidence_target,
        initial_trials=resolved_trials,
    )
    confidence_inputs["replay_triggered"] = False
    confidence_inputs["replay_reason"] = "disabled_fixed_trials"
    confidence_inputs["replay_extra_trials"] = 0
    confidence_inputs["replay_outcome"] = "disabled"

    # ── Output ────────────────────────────────────────────────────────────────
    print_results_table(results, profile=profile, w_tps=w_tps, w_ttft=w_ttft, w_ppl=w_ppl)
    if mode == "depth_profile":
        _print_depth_profile_tables(depth_profile_payload)
    if stability and stability.confidence == "low":
        console.print(
            f"[yellow]Low winner confidence:[/yellow] top-2 gap={stability.top2_gap_abs:.2f} "
            f"({stability.top2_gap_pct:.2f}%)."
        )

    # Baseline compare against user-provided current config
    baseline_cmp = None
    baseline_err = None
    if baseline_config:
        baseline_cmp = _baseline_compare(results, baseline_config)
        if baseline_cmp and baseline_cmp.get("found"):
            tail = ""
            if baseline_cmp.get("delta_tps_p95") is not None or baseline_cmp.get("delta_ttft_p95_ms") is not None:
                tail = (
                    f"  TPS p95 Δ={baseline_cmp.get('delta_tps_p95', 0.0):+.2f}  "
                    f"TTFT p95 Δ={baseline_cmp.get('delta_ttft_p95_ms', 0.0):+.1f}ms"
                )
            console.print(
                f"[dim]Baseline compare:[/dim] score Δ={baseline_cmp['delta_score']:+.2f}  "
                f"TPS Δ={baseline_cmp['delta_tps']:+.2f}  TTFT Δ={baseline_cmp['delta_ttft_ms']:+.1f}ms  "
                f"PPL Δ={baseline_cmp['delta_ppl']:+.2f}{tail}"
            )
        else:
            baseline_err = "baseline config not found in this grid"
            console.print("[yellow]Baseline config not found in this grid.[/yellow]")
    else:
        baseline_cmp = _auto_baseline_compare(results)
        if baseline_cmp and baseline_cmp.get("found"):
            tail = ""
            if baseline_cmp.get("delta_tps_p95") is not None or baseline_cmp.get("delta_ttft_p95_ms") is not None:
                tail = (
                    f"  TPS p95 Δ={baseline_cmp.get('delta_tps_p95', 0.0):+.2f}  "
                    f"TTFT p95 Δ={baseline_cmp.get('delta_ttft_p95_ms', 0.0):+.1f}ms"
                )
            console.print(
                f"[dim]Auto baseline compare ({baseline_cmp.get('baseline_source')}):[/dim] "
                f"score Δ={baseline_cmp['delta_score']:+.2f}  "
                f"TPS Δ={baseline_cmp['delta_tps']:+.2f}  TTFT Δ={baseline_cmp['delta_ttft_ms']:+.1f}ms  "
                f"PPL Δ={baseline_cmp['delta_ppl']:+.2f}{tail}"
            )

    # Optional lightweight agent smoke checks
    smoke_payload = None
    if agent_smoke:
        smoke_payload = _maybe_run_agent_smoke(
            backend=backend,
            engine=engine,
            hardware=hardware,
            model_repo=repo_id,
            results=results,
            models=models,
        )
        if smoke_payload is None:
            console.print("[yellow]Agent smoke skipped: currently supported for llama.cpp on local/modal paths.[/yellow]")
        else:
            console.print(
                f"[dim]Agent smoke:[/dim] {smoke_payload.get('passed')}/{smoke_payload.get('total')} "
                f"({float(smoke_payload.get('pass_rate', 0.0)) * 100:.1f}%)"
            )
            if smoke_payload.get("diagnosis") or smoke_payload.get("status"):
                console.print(
                    f"[dim]Agent smoke diagnosis:[/dim] {smoke_payload.get('diagnosis')}  "
                    f"[dim]status:[/dim] {smoke_payload.get('status')}"
                )

    # Export bundle (JSON + Markdown + frontier SVG)
    prompt_source = "default"
    prompt_chars = None
    prompt_sha12 = None
    prompt_tokens_est = None
    prompt_path = (os.environ.get("SIGILANT_BENCH_PROMPT_FILE", "") or "").strip()
    if prompt_path:
        try:
            txt = Path(prompt_path).read_text(encoding="utf-8")
            prompt_source = prompt_path
            prompt_chars = len(txt)
            prompt_sha12 = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12]
            # Lightweight estimate for quick sanity checks (exact tokenization depends on tokenizer).
            prompt_tokens_est = max(1, int(round(len(txt) / 4.0)))
        except Exception:
            prompt_source = f"{prompt_path} (unreadable)"
    if prompt_tokens_est is not None:
        max_ctx = max((r.config.context for r in results), default=None)
        if max_ctx and prompt_tokens_est < int(0.5 * max_ctx):
            console.print(
                f"[yellow]Prompt depth warning:[/yellow] est prompt tokens ~{prompt_tokens_est}, "
                f"max tested ctx={max_ctx}. Context stress may be weak."
            )

    repro_cmd = build_repro_command(
        model=model,
        backend=backend,
        engine=engine,
        hardware=hardware,
        params_b=params_b,
        max_configs=max_configs,
        trials=resolved_trials,
        score_profile=profile,
    )
    ctx = {
        "model": model,
        "model_label": model_label,
        "backend": backend,
        "engine": engine,
        "hardware": hw_label,
        "params_b": effective_params_b,
        "trials": resolved_trials,
        "confidence_target": confidence_target,
        "score_profile": profile,
        "benchmark_mode": mode,
        "benchmark_prompt_source": prompt_source,
        "benchmark_prompt_chars": prompt_chars,
        "benchmark_prompt_sha12": prompt_sha12,
        "benchmark_prompt_tokens_est": prompt_tokens_est,
        "repro_command": repro_cmd,
    }
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("artifacts") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle = export_bundle(
        results=results,
        path_json=str(run_dir / "sigilant_results.json"),
        path_md=str(run_dir / "sigilant_summary.md"),
        path_svg=str(run_dir / "sigilant_frontier.svg"),
        path_terminal_txt=str(run_dir / "sigilant_terminal.txt"),
        context=ctx,
        stability=stability,
        baseline_compare=baseline_cmp,
        agent_smoke=smoke_payload,
        confidence_inputs=confidence_inputs,
        depth_profile=depth_profile_payload,
    )
    # Keep latest convenience files at repo root while preserving per-run history.
    latest_json = Path("sigilant_results.json")
    latest_md = Path("sigilant_summary.md")
    latest_svg = Path("sigilant_frontier.svg")
    latest_txt = Path("sigilant_terminal.txt")
    latest_json.write_text(Path(bundle["json"]).read_text())
    latest_md.write_text(Path(bundle["md"]).read_text())
    latest_svg.write_text(Path(bundle["svg"]).read_text())
    latest_txt.write_text(Path(bundle["terminal"]).read_text())
    console.print(
        f"[dim]Artifacts:[/dim] {bundle['json']}, {bundle['md']}, {bundle['svg']}, {bundle['terminal']}"
    )
    _print_quick_wow(results=results, baseline_cmp=baseline_cmp, smoke_payload=smoke_payload)

    # keep explicit --json behavior for compatibility messaging
    if json_out:
        path = export_json(results)
        console.print(f"[dim]Legacy JSON also written to {path}[/dim]\n")

    # CI check mode: compare winner against saved baseline
    winner = _best_result(results)
    if update_baseline and winner is not None:
        _write_baseline_file(baseline_file, winner.config.label())
        console.print(f"[dim]Baseline updated:[/dim] {baseline_file}")
    if check:
        if baseline_config and baseline_err:
            console.print(f"[red]CHECK FAIL[/red] {baseline_err}")
            raise typer.Exit(2)
        ok, msg = _check_against_baseline(baseline_file, winner.config.label() if winner else None)
        if ok:
            console.print(f"[green]CHECK PASS[/green] {msg}")
        else:
            console.print(f"[red]CHECK FAIL[/red] {msg}")
            raise typer.Exit(2)


@app.command()
def info() -> None:
    """Show detected local hardware and installed engines."""
    from .core.hardware import detect_hardware

    hw = detect_hardware()

    console.print()
    console.print(f"[bold]GPU:[/bold]       {hw.gpu_name}")
    console.print(f"[bold]VRAM:[/bold]      {hw.vram_gb} GB")
    console.print(f"[bold]Compute:[/bold]   {hw.compute_backend}")
    console.print(f"[bold]Platform:[/bold]  {hw.os}")
    console.print()
    console.print("[bold]Engines (llama.cpp):[/bold]")
    from .engines.llama_cli_engine import find_binary
    binary = find_binary()
    if binary:
        console.print(f"  [green]✓[/green]  llama-cli binary  [dim]({binary})[/dim]")
    else:
        console.print(f"  [dim]–  llama-cli binary not found[/dim]")
        console.print(f"     [dim]Set SIGILANT_LLAMA_CLI=/path/to/llama-cli  or build llama.cpp[/dim]")
    _print_engine_status("llama_cpp", "llama-cpp-python", "pip install 'sigilant-runner[llama]'")
    console.print()
    console.print("[bold]Cloud backend:[/bold]")
    _print_engine_status("modal",  "Modal",  "pip install 'sigilant-runner[modal]'")
    console.print()


@app.callback(invoke_without_command=True)
def version_flag(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", is_eager=True, help="Show version and exit."),
) -> None:
    if version:
        console.print(f"sigilant-runner {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


# ── helpers ───────────────────────────────────────────────────────────────────

def _dispatch(backend: str, engine: str, grid, trials: int = 1) -> list:
    if backend == "local":
        from .backends.local import LocalBackend
        return LocalBackend(engine=engine, trials=trials).run(grid)
    elif backend == "modal":
        from .backends.modal_backend import ModalBackend
        return ModalBackend(engine=engine, trials=trials).run(grid)
    else:
        raise RuntimeError(f"Unknown backend: {backend!r}. Choose local | modal")


def _print_engine_status(import_name: str, label: str, install: str) -> None:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "")
        ver_str = f"  [dim](v{ver})[/dim]" if ver else ""
        console.print(f"  [green]✓[/green]  {label}{ver_str}")
    except ImportError:
        from rich.markup import escape
        console.print(f"  [dim]–  {label} not installed[/dim]")
        console.print(f"     [dim]{escape(install)}[/dim]")


def _best_result(results):
    from .core.ranking import ranked_succeeded
    ok = ranked_succeeded(results)
    return ok[0] if ok else None


def _print_depth_profile_tables(depth_profile_payload):
    if not isinstance(depth_profile_payload, dict):
        return
    passes = depth_profile_payload.get("passes") or []
    if not passes:
        return

    console.print()
    console.print("[bold]Depth Profile (bucket winners)[/bold]")
    winners = depth_profile_payload.get("bucket_winners") or {}
    if winners:
        console.print(
            f"[dim]best_at_8k:[/dim] {winners.get('best_at_8k') or 'n/a'}  "
            f"[dim]best_at_14k:[/dim] {winners.get('best_at_14k') or 'n/a'}  "
            f"[dim]best_at_28k:[/dim] {winners.get('best_at_28k') or 'n/a'}"
        )
    for p in passes:
        dlabel = str(p.get("depth_label") or "unknown")
        rows = p.get("results") or []
        table = Table(
            title=f"{dlabel} prompt pass",
            box=box.SIMPLE_HEAD,
            show_edge=False,
            pad_edge=False,
        )
        table.add_column("Config", min_width=42)
        table.add_column("Status", style="dim")
        table.add_column("TPS", justify="right", style="cyan")
        table.add_column("TTFT", justify="right", style="cyan")
        table.add_column("PPL", justify="right", style="cyan")
        table.add_column("Score", justify="right", style="bold")

        for r in rows:
            status = str(r.get("status") or "failed")
            tps = r.get("tps")
            ttft = r.get("ttft_ms")
            ppl = r.get("ppl")
            score = r.get("score")
            table.add_row(
                str(r.get("label") or "unknown"),
                status,
                (f"{float(tps):.1f}" if tps is not None else "—"),
                (f"{float(ttft):.0f}ms" if ttft is not None else "—"),
                (f"{float(ppl):.2f}" if ppl is not None else "—"),
                (str(int(score)) if score is not None else "—"),
            )
        console.print(table)


def _baseline_compare(results, baseline_config: str):
    def _r(v, n=2):
        return round(float(v), n)
    bits = [x.strip() for x in (baseline_config or "").split(",")]
    if len(bits) != 4:
        return {"found": False, "error": "invalid_format"}
    q, ctx_s, kv, regime = bits
    try:
        ctx = int(ctx_s)
    except Exception:
        return {"found": False, "error": "invalid_ctx"}
    ok = [r for r in results if r.succeeded and r.score is not None]
    if not ok:
        return {"found": False, "error": "no_results"}
    winner = sorted(ok, key=lambda r: r.score or 0, reverse=True)[0]
    base = None
    for r in ok:
        c = r.config
        if c.quant_label.lower() == q.lower() and int(c.context) == int(ctx) and c.kv_type.lower() == kv.lower() and c.regime.lower() == regime.lower():
            base = r
            break
    if base is None:
        return {"found": False}
    return {
        "found": True,
        "baseline_source": "user",
        "baseline_label": base.config.label(),
        "winner_label": winner.config.label(),
        "delta_score": _r((winner.score or 0) - (base.score or 0), 2),
        "delta_tps": _r((winner.tps or 0) - (base.tps or 0), 2),
        "delta_tps_p95": (
            _r((winner.tps_p95 or 0) - (base.tps_p95 or 0), 2)
            if (winner.tps_p95 is not None and base.tps_p95 is not None) else None
        ),
        "delta_ttft_ms": _r((winner.ttft_ms or 0) - (base.ttft_ms or 0), 1),
        "delta_ttft_p95_ms": (
            _r((winner.ttft_p95_ms or 0) - (base.ttft_p95_ms or 0), 1)
            if (winner.ttft_p95_ms is not None and base.ttft_p95_ms is not None) else None
        ),
        "delta_ppl": _r((winner.ppl or 0) - (base.ppl or 0), 2) if (winner.ppl is not None and base.ppl is not None) else 0.0,
    }


def _quant_bits(label: str) -> int:
    s = (label or "").strip().lower()
    if "fp16_baseline" in s:
        return 16
    if "int8_w8a8" in s:
        return 8
    if "awq4_marlin" in s or "gptq4_marlin" in s:
        return 4
    if "f16" in s or "fp16" in s or "bf16" in s:
        return 16
    if "q8" in s:
        return 8
    if "q6" in s:
        return 6
    if "q5" in s:
        return 5
    if "q4" in s:
        return 4
    if "iq3" in s or "q3" in s:
        return 3
    if "q2" in s:
        return 2
    return 0


def _auto_baseline_compare(results):
    def _r(v, n=2):
        return round(float(v), n)
    ok = [r for r in results if r.succeeded and r.score is not None]
    if not ok:
        return {"found": False, "error": "no_results"}
    winner = sorted(ok, key=lambda r: r.score or 0, reverse=True)[0]
    # Highest-precision quant that actually fit and completed.
    max_bits = max((_quant_bits(r.config.quant_label) for r in ok), default=0)
    pool = [r for r in ok if _quant_bits(r.config.quant_label) == max_bits]
    # Choose strongest baseline within highest-precision quant.
    base = sorted(pool, key=lambda r: r.score or 0, reverse=True)[0] if pool else None
    if base is None:
        return {"found": False, "error": "no_baseline_candidate"}
    return {
        "found": True,
        "baseline_source": f"auto:max_precision({base.config.quant_label})",
        "baseline_label": base.config.label(),
        "winner_label": winner.config.label(),
        "delta_score": _r((winner.score or 0) - (base.score or 0), 2),
        "delta_tps": _r((winner.tps or 0) - (base.tps or 0), 2),
        "delta_tps_p95": (
            _r((winner.tps_p95 or 0) - (base.tps_p95 or 0), 2)
            if (winner.tps_p95 is not None and base.tps_p95 is not None) else None
        ),
        "delta_ttft_ms": _r((winner.ttft_ms or 0) - (base.ttft_ms or 0), 1),
        "delta_ttft_p95_ms": (
            _r((winner.ttft_p95_ms or 0) - (base.ttft_p95_ms or 0), 1)
            if (winner.ttft_p95_ms is not None and base.ttft_p95_ms is not None) else None
        ),
        "delta_ppl": _r((winner.ppl or 0) - (base.ppl or 0), 2) if (winner.ppl is not None and base.ppl is not None) else 0.0,
    }


def _write_baseline_file(path: str, winner_label: str) -> None:
    payload = {"schema": "sigilant.runner.baseline.v1", "winner_label": winner_label}
    Path(path).write_text(json.dumps(payload, indent=2))


def _check_against_baseline(path: str, winner_label: Optional[str]):
    if not winner_label:
        return False, "no winner result available"
    p = Path(path)
    if not p.exists():
        return False, f"baseline file missing: {path}"
    try:
        payload = json.loads(p.read_text())
        expected = str(payload.get("winner_label") or "").strip()
    except Exception as exc:
        return False, f"invalid baseline file: {type(exc).__name__}: {exc}"
    if not expected:
        return False, "baseline winner_label is empty"
    if expected == winner_label:
        return True, f"winner unchanged ({winner_label})"
    return False, f"winner drifted expected='{expected}' got='{winner_label}'"


def _maybe_run_agent_smoke(*, backend: str, engine: str, hardware: str, model_repo: str, results, models):
    if engine != "llama.cpp":
        return None
    winner = _best_result(results)
    if winner is None:
        return None
    quant = winner.config.quant_label.upper()
    model_ref = None
    for q, ref in models:
        if str(q).upper() == quant:
            model_ref = ref
            break
    if not model_ref:
        return None
    if backend == "local":
        from .engines.llama_cli_engine import find_binary
        exe = find_binary()
        if not exe:
            return None
        from .core.agent_smoke import run_agent_smoke
        return run_agent_smoke(
            llama_cli=exe,
            model_path=model_ref,
            ctx=winner.config.context,
            kv_type=winner.config.kv_type,
        )
    if backend == "modal":
        if not model_repo:
            return None
        from .backends.modal_backend import ModalBackend
        mb = ModalBackend(hardware=hardware, engine=engine, trials=1)
        return mb.run_agent_smoke(
            quant_label=winner.config.quant_label,
            context=winner.config.context,
            kv_type=winner.config.kv_type,
            model_repo=model_repo,
            model_filename=model_ref,
        )
    return None


def _print_quick_wow(*, results, baseline_cmp, smoke_payload):
    winner = _best_result(results)
    if winner is None:
        return
    console.print()
    console.print("[bold]Quick wow[/bold]")
    console.print(f"- Recommended: {winner.config.label()}")
    console.print(
        f"- Core metrics: TPS {winner.tps:.1f}, TTFT {winner.ttft_ms:.0f}ms, "
        f"PPL {winner.ppl if winner.ppl is not None else '—'}"
    )
    if baseline_cmp and baseline_cmp.get("found"):
        console.print(
            "- Baseline delta: "
            f"score {baseline_cmp.get('delta_score', 0):+.2f}, "
            f"TPS {baseline_cmp.get('delta_tps', 0):+.2f}, "
            f"TTFT {baseline_cmp.get('delta_ttft_ms', 0):+.1f}ms"
        )
    if smoke_payload:
        console.print(
            "- Agent smoke: "
            f"{smoke_payload.get('passed')}/{smoke_payload.get('total')} "
            f"({float(smoke_payload.get('pass_rate', 0.0)) * 100:.1f}%) "
            f"[{smoke_payload.get('diagnosis')}]"
        )
    console.print("- Next step: apply this config directly in your serving launch args.")


def _result_key(r) -> Tuple[str, int, str, str]:
    c = r.config
    return (str(c.quant_label), int(c.context), str(c.kv_type), str(c.regime))


def _topn_configs(results, n: int):
    from .core.ranking import ranked_succeeded
    ok = ranked_succeeded(results)
    return [r.config for r in ok[:max(0, int(n))]]


def _merge_replay_results(results, replay_rows):
    rep_map = {_result_key(r): r for r in replay_rows if r.succeeded}
    out = []
    for r in results:
        out.append(rep_map.get(_result_key(r), r))
    return out


def _top2_variance_proxy(results) -> Optional[float]:
    from .core.ranking import ranked_succeeded
    ok = ranked_succeeded(results)[:2]
    if len(ok) < 2:
        return None
    vals = []
    for r in ok:
        chunks = []
        if r.tps_p95 and r.tps:
            chunks.append(abs(float(r.tps_p95) - float(r.tps)) / max(float(r.tps), 1e-6) * 100.0)
        if r.ttft_p95_ms and r.ttft_ms:
            chunks.append(abs(float(r.ttft_p95_ms) - float(r.ttft_ms)) / max(float(r.ttft_ms), 1e-6) * 100.0)
        if chunks:
            vals.append(sum(chunks) / len(chunks))
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def _build_confidence_inputs(*, results, stability, confidence_target: str, initial_trials: int) -> Dict[str, Any]:
    return {
        "confidence_target": confidence_target,
        "trials_initial": int(initial_trials),
        "gap_abs_before": (stability.top2_gap_abs if stability else None),
        "gap_pct_before": (stability.top2_gap_pct if stability else None),
        "variance_pct_before": _top2_variance_proxy(results),
        "replay_triggered": False,
        "replay_reason": None,
        "replay_extra_trials": 0,
        "replay_outcome": "not_needed",
        "gap_pct_after": (stability.top2_gap_pct if stability else None),
        "variance_pct_after": _top2_variance_proxy(results),
    }
