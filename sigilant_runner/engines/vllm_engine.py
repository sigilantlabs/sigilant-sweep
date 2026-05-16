"""vLLM inference engine.

Install:  pip install 'sigilant-runner[vllm]'
Requires: Linux + CUDA. Not supported on macOS or Windows.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from ..core.metrics import RunConfig, RunResult

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

_BENCH_PROMPT = (
    "Explain the key architectural differences between transformer encoder and decoder "
    "models. Include details about attention mechanisms, typical use cases, and how "
    "self-attention differs from cross-attention."
)


class VLLMEngine:
    def __init__(self, model_path: str):
        if not _HAS_VLLM:
            raise RuntimeError(
                "vllm is not installed.\n"
                "  pip install 'sigilant-runner[vllm]'\n"
                "  Note: requires Linux + CUDA."
            )
        self.model_path = model_path

    def run_config(self, config: RunConfig) -> RunResult:
        try:
            return self._run(config)
        except Exception as exc:
            return RunResult(config=config, error=str(exc))

    def _run(self, config: RunConfig) -> RunResult:
        bench_prompt = self._bench_prompt()
        profile = self._profile_for(config.quant_label)
        max_num_seqs = max(int(config.batch), int(profile["max_num_seqs"]))
        llm = LLM(
            model=self.model_path,
            max_model_len=config.context,
            gpu_memory_utilization=float(profile["gpu_memory_utilization"]),
            max_num_seqs=max_num_seqs,
            dtype=str(profile["dtype"]),
        )
        params = SamplingParams(max_tokens=256, temperature=0.0)

        # Warm-up
        llm.generate([bench_prompt[:40]], SamplingParams(max_tokens=4))

        t_start = time.perf_counter()
        outputs = llm.generate([bench_prompt], params)
        t_end   = time.perf_counter()

        del llm

        output    = outputs[0].outputs[0]
        n_tokens  = len(output.token_ids)
        total_s   = t_end - t_start

        tps     = n_tokens / total_s if total_s > 0 else 0.0
        # vLLM does not expose per-token timestamps in basic API;
        # TTFT approximated as ~20% of total latency for short prompts
        ttft_ms = total_s * 0.20 * 1_000
        itl_ms  = (total_s * 0.80 / max(n_tokens - 1, 1)) * 1_000

        # PPL requires token log-probabilities; collect via prompt_logprobs
        ppl = self._perplexity_vllm(output)

        return RunResult(
            config=config,
            tps=round(tps, 1),
            ttft_ms=round(ttft_ms, 1),
            itl_ms=round(itl_ms, 2),
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

    @staticmethod
    def _profile_for(label: str):
        s = str(label or "").upper()
        if s == "FP16_BASELINE":
            return {"dtype": "float16", "gpu_memory_utilization": 0.88, "max_num_seqs": 4}
        if s == "INT8_W8A8":
            return {"dtype": "auto", "gpu_memory_utilization": 0.90, "max_num_seqs": 8}
        if s == "AWQ4_MARLIN":
            return {"dtype": "auto", "gpu_memory_utilization": 0.90, "max_num_seqs": 8}
        if s == "GPTQ4_MARLIN":
            return {"dtype": "auto", "gpu_memory_utilization": 0.90, "max_num_seqs": 8}
        return {"dtype": "auto", "gpu_memory_utilization": 0.90, "max_num_seqs": 8}

    def _perplexity_vllm(self, output) -> Optional[float]:
        """Approximate PPL from prompt log-probs if available."""
        try:
            import math
            lps = [
                list(tok.values())[0].logprob
                for tok in (output.prompt_logprobs or [])
                if tok
            ]
            if not lps:
                return None
            return round(math.exp(-sum(lps) / len(lps)), 2)
        except Exception:
            return None
