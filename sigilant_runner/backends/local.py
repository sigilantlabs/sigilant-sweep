from __future__ import annotations

from typing import Callable, List, Optional

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ..core.metrics import RunConfig, RunResult


class LocalBackend:
    def __init__(self, engine: str = "llama.cpp", trials: int = 1):
        self.engine_name = engine
        self.trials = trials

    def run(
        self,
        configs: List[RunConfig],
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[RunResult]:
        engine = self._make_engine(configs)
        results: List[RunResult] = []

        def _median(vals):
            s = sorted(vals)
            n = len(s)
            return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

        def _percentile(vals, p: float):
            if not vals:
                return None
            s = sorted(vals)
            if len(s) == 1:
                return float(s[0])
            pos = (len(s) - 1) * max(0.0, min(1.0, float(p)))
            lo = int(pos)
            hi = min(lo + 1, len(s) - 1)
            frac = pos - lo
            return float(s[lo] * (1.0 - frac) + s[hi] * frac)

        def _trial_starts(n_cfg: int, n_trials: int) -> List[int]:
            n_trials = max(1, int(n_trials))
            if n_cfg <= 0:
                return [0] * n_trials
            # Matches desired pattern:
            # n=16,t=3 -> starts 0,5,10 ; n=16,t=8 -> 0,2,4,6,...
            stride = max(1, n_cfg // n_trials)
            return [((t * stride) % n_cfg) for t in range(n_trials)]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("", total=len(configs))
            trial_count = max(1, int(self.trials))
            starts = _trial_starts(len(configs), trial_count)
            buckets: List[List[RunResult]] = [[] for _ in configs]
            errors: List[Optional[str]] = [None for _ in configs]

            # Trial-wise rotated execution to reduce order bias.
            for t in range(trial_count):
                start = starts[t]
                for off in range(len(configs)):
                    i = (start + off) % len(configs)
                    cfg = configs[i]
                    progress.update(
                        task,
                        description=f"[cyan]trial {t+1}/{trial_count} · {i+1}/{len(configs)}[/cyan] {cfg.label()}",
                        completed=i,
                    )
                    rr = engine.run_config(cfg)
                    if rr.succeeded:
                        buckets[i].append(rr)
                    else:
                        errors[i] = rr.error
                    if progress_cb:
                        progress_cb(i + 1, len(configs), cfg.label())

            # Aggregate per-config across trials (p50 + optional p95).
            for i, cfg in enumerate(configs):
                trial_rows = buckets[i]
                last_err = errors[i]
                if not trial_rows:
                    results.append(RunResult(config=cfg, error=last_err or "all trials failed"))
                else:
                    tps_vals = [r.tps for r in trial_rows]
                    ttft_vals = [r.ttft_ms for r in trial_rows]
                    itl_vals = [r.itl_ms for r in trial_rows]
                    out = RunResult(
                        config=cfg,
                        tps=round(_median(tps_vals), 1),
                        ttft_ms=round(_median(ttft_vals), 1),
                        itl_ms=round(_median(itl_vals), 2),
                        ppl=trial_rows[0].ppl,
                        tps_p95=round(_percentile(tps_vals, 0.95), 1) if len(tps_vals) >= 4 else None,
                        ttft_p95_ms=round(_percentile(ttft_vals, 0.95), 1) if len(ttft_vals) >= 4 else None,
                    )
                    results.append(out)
                if progress_cb:
                    progress_cb(i + 1, len(configs), cfg.label())

        return results

    def _make_engine(self, configs: List[RunConfig]):
        if self.engine_name == "vllm":
            from ..engines.vllm_engine import VLLMEngine
            return _MultiFileEngine(VLLMEngine, configs)

        if self.engine_name == "llama.cpp":
            return self._make_llama_engine(configs)

        raise ValueError(f"Unknown engine: {self.engine_name!r}")

    @staticmethod
    def _make_llama_engine(configs: List[RunConfig]):
        # Prefer the llama-cli binary (no Python compilation needed).
        # Fall back to llama-cpp-python if the binary is not found.
        from ..engines.llama_cli_engine import find_binary
        binary = find_binary()
        if binary:
            from ..engines.llama_cli_engine import LlamaCLIEngine
            return _MultiFileEngine(lambda path: LlamaCLIEngine(path, binary=binary), configs)

        try:
            from ..engines.llama_engine import LlamaEngine
            return _MultiFileEngine(LlamaEngine, configs)
        except ImportError:
            raise RuntimeError(
                "No llama.cpp engine available.\n"
                "  Option A — set SIGILANT_LLAMA_CLI=/path/to/llama-cli\n"
                "  Option B — pip install 'sigilant-runner[llama]'"
            )


class _MultiFileEngine:
    """Wrapper that creates a new engine instance per config (model path may differ)."""

    def __init__(self, engine_cls, configs: List[RunConfig]):
        self._cls = engine_cls
        self._cache: dict[str, object] = {}

    def run_config(self, config: RunConfig) -> RunResult:
        path = config.model_path
        if path not in self._cache:
            self._cache.clear()   # release previous model's VRAM
            self._cache[path] = self._cls(path)
        return self._cache[path].run_config(config)
