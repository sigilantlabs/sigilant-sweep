"""llama-cli binary engine.

Uses an existing llama-cli binary instead of llama-cpp-python.
Auto-detected from PATH and common install locations, or override via:
  export SIGILANT_LLAMA_CLI=/path/to/llama-cli

Timing is parsed directly from llama-cli's built-in output:
  llama_print_timings: prompt eval time = X ms / N tokens  → TTFT
  llama_print_timings:        eval time = X ms / N tokens  → TPS, ITL
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ..core.metrics import RunConfig, RunResult

_BENCH_PROMPT = (
    "Explain the difference between transformer encoder and decoder models."
)

_PPL_CORPUS = (
    "The transformer architecture has revolutionized natural language processing. "
    "Self-attention mechanisms allow models to weigh the importance of different words "
    "in a sequence when producing representations. Large language models trained on "
    "diverse corpora demonstrate emergent capabilities including in-context learning, "
    "chain-of-thought reasoning, and instruction following. Quantization reduces model "
    "precision to decrease memory footprint, with quality loss scaling as bit-width "
    "decreases. The trade-off between inference speed and output fidelity depends "
    "on model architecture, quantization scheme, and deployment context window."
)

_SEARCH_PATHS = [
    os.environ.get("SIGILANT_LLAMA_CLI", ""),
    "llama-cli",                                     # on PATH
    "/usr/local/bin/llama-cli",
    str(Path.home() / "llama.cpp/build/bin/llama-cli"),
    "/opt/homebrew/bin/llama-cli",
]


def find_binary() -> Optional[str]:
    for candidate in _SEARCH_PATHS:
        if not candidate:
            continue
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True, timeout=5,
            )
            if result.returncode in (0, 1):   # llama-cli --version may exit 1
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


class LlamaCLIEngine:
    def __init__(self, model_path: str, binary: Optional[str] = None):
        self.model_path = model_path
        self.binary = binary or find_binary()
        if not self.binary:
            raise RuntimeError(
                "llama-cli binary not found.\n"
                "Set SIGILANT_LLAMA_CLI=/path/to/llama-cli  or install llama-cpp-python:\n"
                "  pip install 'sigilant-runner[llama]'"
            )

    def run_config(self, config: RunConfig) -> RunResult:
        try:
            return self._run(config)
        except Exception as exc:
            return RunResult(config=config, error=str(exc))

    def _run(self, config: RunConfig) -> RunResult:
        kv_args = self._kv_args(config.kv_type)
        bench_prompt = self._bench_prompt()

        cmd = [
            self.binary,
            "-m", self.model_path,
            "-p", bench_prompt,
            "--single-turn",
            "--temp", "0",
            "--top-k", "1",
            "--seed", "42",
            "-t", "6",
            "-n", "256",
            "--ctx-size", str(config.context),
            "--simple-io",
            *kv_args,
        ]

        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            # keep this generous for weaker local machines
        )

        # llama-cli writes timing to stderr
        timing = out.stderr + out.stdout
        tps, ttft_ms, itl_ms = self._parse_timings(timing)
        ppl = self._compute_ppl(config)

        return RunResult(
            config=config,
            tps=tps,
            ttft_ms=ttft_ms,
            itl_ms=itl_ms,
            ppl=ppl,
        )

    @staticmethod
    def _bench_prompt() -> str:
        p = (os.environ.get("SIGILANT_BENCH_PROMPT_FILE", "") or "").strip()
        if not p:
            return _BENCH_PROMPT
        try:
            t = Path(p).read_text(encoding="utf-8").strip()
            return t or _BENCH_PROMPT
        except Exception:
            return _BENCH_PROMPT

    def _compute_ppl(self, config: RunConfig) -> Optional[float]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(_PPL_CORPUS)
            corpus_path = f.name

        try:
            cmd = [
                self.binary,
                "-m", self.model_path,
                "--perplexity",
                "--file", corpus_path,
                "--ctx-size", "512",
                "--no-display-prompt",
            ]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return self._parse_ppl(out.stderr + out.stdout)
        except Exception:
            return None
        finally:
            Path(corpus_path).unlink(missing_ok=True)

    # ── parsers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_timings(output: str) -> tuple[float, float, float]:
        """Extract TPS, TTFT, and ITL from llama_print_timings output."""
        tps = ttft_ms = itl_ms = 0.0

        # prompt eval time =  456.78 ms /  32 tokens  (14.27 ms per token,  70.06 tokens per second)
        m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms", output)
        if m:
            ttft_ms = round(float(m.group(1)), 1)

        # eval time =  5678.90 ms /  199 runs  (28.54 ms per token,  35.05 tokens per second)
        m = re.search(r"\beval time\s*=\s*[\d.]+\s*ms\s*/\s*\d+\s*runs\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)", output)
        if m:
            itl_ms = round(float(m.group(1)), 2)
            tps    = round(float(m.group(2)), 1)

        return tps, ttft_ms, itl_ms

    @staticmethod
    def _parse_ppl(output: str) -> Optional[float]:
        patterns = [
            r"Final estimate:\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)",
            r"\bPPL\s*=\s*([0-9]+(?:\.[0-9]+)?)\b",
            r"\bperplexity\s*:\s*([0-9]+(?:\.[0-9]+)?)\b",
            r"\bperplexity\s*=\s*([0-9]+(?:\.[0-9]+)?)\b",
        ]
        for pat in patterns:
            m = re.search(pat, output, re.IGNORECASE)
            if m:
                return round(float(m.group(1)), 2)
        return None

    @staticmethod
    def _kv_args(kv_type: str) -> list[str]:
        if kv_type == "k8v8":
            return ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
        return []   # k16v16 is the default (fp16)
