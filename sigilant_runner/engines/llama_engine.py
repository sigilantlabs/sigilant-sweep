"""llama-cpp-python inference engine.

Install:  pip install 'sigilant-runner[llama]'
For CUDA: CMAKE_ARGS="-DGGML_CUDA=on" pip install 'sigilant-runner[llama]'
"""
from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Optional

from ..core.metrics import RunConfig, RunResult

try:
    from llama_cpp import Llama
    _HAS_LLAMA = True
except ImportError:
    _HAS_LLAMA = False

# Fixed prompt — same across all configs for fair comparison
_BENCH_PROMPT = (
    "Explain the key architectural differences between transformer encoder and decoder "
    "models. Include details about attention mechanisms, typical use cases, and how "
    "self-attention differs from cross-attention."
)

# Small fixed corpus used for perplexity measurement
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


class LlamaEngine:
    def __init__(self, model_path: str):
        if not _HAS_LLAMA:
            raise RuntimeError(
                "llama-cpp-python is not installed.\n"
                "  pip install 'sigilant-runner[llama]'\n"
                "  For CUDA: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install 'sigilant-runner[llama]'"
            )
        self.model_path = model_path

    def run_config(self, config: RunConfig) -> RunResult:
        try:
            return self._run(config)
        except Exception as exc:
            return RunResult(config=config, error=str(exc))

    def _run(self, config: RunConfig) -> RunResult:
        llm = Llama(
            model_path=self.model_path,
            n_ctx=config.context,
            n_batch=config.batch,
            n_gpu_layers=-1,    # offload all layers to GPU
            verbose=False,
        )

        # Warm-up: short prompt to prime GPU and caches
        list(llm.create_completion(_BENCH_PROMPT[:40], max_tokens=4, temperature=0.0, stream=True))

        # Timed inference
        t_start = time.perf_counter()
        t_first: Optional[float] = None
        n_tokens = 0

        for chunk in llm.create_completion(
            _BENCH_PROMPT,
            max_tokens=256,
            temperature=0.0,
            stream=True,
        ):
            if t_first is None:
                t_first = time.perf_counter()
            n_tokens += 1

        t_end = time.perf_counter()

        ttft_ms  = (t_first - t_start) * 1_000 if t_first else 0.0
        gen_s    = t_end - (t_first or t_start)
        tps      = n_tokens / gen_s if gen_s > 0 else 0.0
        itl_ms   = (gen_s / max(n_tokens - 1, 1)) * 1_000

        ppl = self._perplexity(llm)

        del llm   # release VRAM before next config

        return RunResult(
            config=config,
            tps=round(tps, 1),
            ttft_ms=round(ttft_ms, 1),
            itl_ms=round(itl_ms, 2),
            ppl=round(ppl, 2) if ppl is not None else None,
        )

    def _perplexity(self, llm: "Llama") -> Optional[float]:
        """Compute PPL on the fixed corpus using token log-probabilities."""
        try:
            result = llm.create_completion(
                _PPL_CORPUS,
                max_tokens=1,
                temperature=0.0,
                logprobs=1,
                echo=True,
            )
            token_logprobs = result["choices"][0]["logprobs"]["token_logprobs"]
            valid = [lp for lp in token_logprobs if lp is not None]
            if not valid:
                return None
            return round(math.exp(-sum(valid) / len(valid)), 2)
        except Exception:
            return None
