"""Interactive credential check and setup walkthrough for all backends."""
from __future__ import annotations

import os
import subprocess
import webbrowser
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

_TICK  = "[bold green]✓[/bold green]"
_CROSS = "[bold red]✗[/bold red]"
_WARN  = "[bold yellow]⚠[/bold yellow]"


def run_setup() -> None:
    console.print()
    console.print(Panel.fit(
        "[bold cyan]sigilant-runner setup[/bold cyan]\n"
        "[dim]Checks hardware and credentials for every backend.[/dim]",
        border_style="cyan",
    ))

    _check_local()
    _check_modal()
    _check_runpod()

    console.print()
    console.print(Rule("[dim]ready[/dim]"))
    console.print()
    console.print("Start a sweep:")
    console.print("  [bold green]sigilant-runner run --model mistralai/Mistral-7B-Instruct-v0.3[/bold green]")
    console.print("  [dim]sigilant-runner run --model ... --backend modal --hardware a10g[/dim]")
    console.print("  [dim]sigilant-runner run --model ... --backend runpod --hardware rtx4090[/dim]")
    console.print()


# ── Local ────────────────────────────────────────────────────────────────────

def _check_local() -> None:
    console.print("\n[bold]Local backend[/bold]")

    from .core.hardware import detect_hardware
    hw = detect_hardware()

    if hw.compute_backend == "cuda":
        console.print(f"  {_TICK} GPU: {hw.gpu_name}  ({hw.vram_gb} GB VRAM, CUDA)")
    elif hw.compute_backend == "metal":
        console.print(f"  {_TICK} GPU: {hw.gpu_name}  ({hw.vram_gb} GB shared, Metal)")
    else:
        console.print(f"  {_WARN} No GPU detected — CPU-only runs will be very slow")

    _check_package("llama_cpp", "llama-cpp-python", "sigilant-runner[llama]",
                   note="For CUDA: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install 'sigilant-runner[llama]'")
    _check_package("vllm", "vLLM", "sigilant-runner[vllm]",
                   note="Linux + CUDA only", optional=True)


def _check_package(import_name: str, display: str, install: str,
                   note: str = "", optional: bool = False) -> bool:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "")
        ver_str = f" (v{ver})" if ver else ""
        console.print(f"  {_TICK} {display} installed{ver_str}")
        return True
    except ImportError:
        if optional:
            console.print(f"  [dim]  {display} not installed (optional)[/dim]")
            console.print(f"  [dim]    pip install '{install}'[/dim]")
            if note:
                console.print(f"  [dim]    {note}[/dim]")
        else:
            console.print(f"  {_WARN} {display} not installed")
            console.print(f"    pip install '{install}'")
            if note:
                console.print(f"    [dim]{note}[/dim]")
        return False


# ── Modal ────────────────────────────────────────────────────────────────────

def _check_modal() -> None:
    console.print("\n[bold]Modal backend[/bold]")

    if not _check_package("modal", "modal SDK", "sigilant-runner[modal]", optional=True):
        return

    modal_toml = Path.home() / ".modal.toml"
    has_env    = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
    has_file   = modal_toml.exists() and modal_toml.stat().st_size > 10

    if has_env:
        console.print(f"  {_TICK} Credentials: MODAL_TOKEN_ID / MODAL_TOKEN_SECRET (env vars)")
        _verify_modal_cli()
    elif has_file:
        console.print(f"  {_TICK} Credentials: ~/.modal.toml")
        _verify_modal_cli()
    else:
        console.print(f"  {_CROSS} No Modal credentials found")
        console.print("    To authenticate:")
        console.print("      1. Create a free account at [link]https://modal.com[/link]")
        console.print("      2. [bold]modal token new[/bold]   (saves to ~/.modal.toml)")
        console.print("         or set MODAL_TOKEN_ID + MODAL_TOKEN_SECRET in your shell")

        if typer.confirm("\n  Walk me through it now?", default=False):
            _modal_guided_setup()


def _verify_modal_cli() -> None:
    try:
        out = subprocess.run(
            ["modal", "profile", "current"],
            capture_output=True, text=True, timeout=8,
        )
        if out.returncode == 0:
            profile = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
            console.print(f"  {_TICK} Modal connection verified" + (f"  [dim]({profile})[/dim]" if profile else ""))
        else:
            console.print(f"  {_WARN} Modal CLI responded with an error — run [bold]modal token new[/bold] to refresh")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        console.print(f"  {_WARN} Could not verify Modal connection (modal CLI not on PATH)")


def _modal_guided_setup() -> None:
    console.print("\n  Opening Modal token settings in your browser...")
    webbrowser.open("https://modal.com/settings/tokens")
    console.print()
    console.print("  After creating a token, run one of:")
    console.print("    [bold]modal token new[/bold]                           (recommended — saves to ~/.modal.toml)")
    console.print("    [dim]export MODAL_TOKEN_ID=<id>[/dim]")
    console.print("    [dim]export MODAL_TOKEN_SECRET=<secret>[/dim]")
    console.print()
    typer.confirm("  Done? Re-check credentials", default=True)
    _check_modal()


# ── RunPod ───────────────────────────────────────────────────────────────────

def _check_runpod() -> None:
    console.print("\n[bold]RunPod backend[/bold]")

    if not _check_package("runpod", "runpod SDK", "sigilant-runner[runpod]", optional=True):
        return

    api_key     = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("SIGILANT_RUNPOD_ENDPOINT_ID")

    if api_key:
        console.print(f"  {_TICK} RUNPOD_API_KEY set")
        _verify_runpod_key(api_key)
    else:
        console.print(f"  {_CROSS} RUNPOD_API_KEY not set")
        console.print("    To authenticate:")
        console.print("      1. Create an account at [link]https://runpod.io[/link]")
        console.print("      2. Settings → API Keys → New API Key")
        console.print("      3. [bold]export RUNPOD_API_KEY=<your-key>[/bold]")

        if typer.confirm("\n  Walk me through it now?", default=False):
            _runpod_guided_setup()
        return

    if endpoint_id:
        console.print(f"  {_TICK} Worker endpoint: {endpoint_id}")
    else:
        console.print(f"  {_WARN} SIGILANT_RUNPOD_ENDPOINT_ID not set — worker not yet deployed")
        console.print("    Deploy the worker once with:")
        console.print("      [bold]sigilant-runner deploy --backend runpod[/bold]")
        console.print("    Then set the printed endpoint ID:")
        console.print("      [dim]export SIGILANT_RUNPOD_ENDPOINT_ID=<endpoint-id>[/dim]")


def _verify_runpod_key(api_key: str) -> None:
    try:
        import runpod as rp
        rp.api_key = api_key
        endpoints  = rp.get_endpoints()
        console.print(f"  {_TICK} RunPod connection verified  [dim]({len(endpoints)} endpoint(s) found)[/dim]")
    except Exception as exc:
        console.print(f"  {_WARN} RunPod connection check failed: {exc}")


def _runpod_guided_setup() -> None:
    console.print("\n  Opening RunPod settings in your browser...")
    webbrowser.open("https://www.runpod.io/console/user/settings")
    console.print()
    console.print("  After generating your API key:")
    console.print("    [bold]export RUNPOD_API_KEY=<your-api-key>[/bold]")
    console.print()
    typer.confirm("  Done? Re-check credentials", default=True)
    _check_runpod()
