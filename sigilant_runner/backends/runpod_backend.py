"""RunPod Serverless backend.

Runs the evaluation on RunPod consumer or datacenter GPUs.
Install:  pip install 'sigilant-sweep[runpod]'
Auth:     export RUNPOD_API_KEY=<your-key>

The RunPod backend requires a pre-deployed serverless worker endpoint.
Set endpoint ID with:
    export SIGILANT_RUNPOD_ENDPOINT_ID=<your-endpoint-id>
"""
from __future__ import annotations

import json
import os
import time
from typing import List

from ..core.metrics import RunConfig, RunResult

try:
    import runpod
    _HAS_RUNPOD = True
except ImportError:
    _HAS_RUNPOD = False

_ENDPOINT_ENV = "SIGILANT_RUNPOD_ENDPOINT_ID"
_POLL_INTERVAL = 5   # seconds between status polls
_JOB_TIMEOUT   = 7200

_GPU_IDS = {
    "rtx4090":   "NVIDIA GeForce RTX 4090",
    "rtx3090":   "NVIDIA GeForce RTX 3090",
    "rtxa6000":  "NVIDIA RTX A6000",
    "a10":       "NVIDIA A10",
    "a100":      "NVIDIA A100 80GB PCIe",
    "h100":      "NVIDIA H100 80GB HBM3",
}


class RunPodBackend:
    def __init__(self, hardware: str = "rtx4090", engine: str = "llama.cpp", trials: int = 1):
        if not _HAS_RUNPOD:
            raise RuntimeError(
                "runpod is not installed.\n"
                "  pip install 'sigilant-sweep[runpod]'"
            )
        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            raise RuntimeError(
                "RUNPOD_API_KEY is not set.\n"
                "  export RUNPOD_API_KEY=<your-key>\n"
                "  Run 'sigilant-sweep setup' for a guided walkthrough."
            )
        runpod.api_key = api_key
        self.hardware  = hardware
        self.engine    = engine
        if self.engine != "llama.cpp":
            raise RuntimeError(
                "RunPod backend currently supports only --engine llama.cpp.\n"
                "Use --backend modal for vLLM runs."
            )

    def run(self, configs: List[RunConfig]) -> List[RunResult]:
        endpoint_id = os.environ.get(_ENDPOINT_ENV)
        if not endpoint_id:
            raise RuntimeError(
                f"{_ENDPOINT_ENV} is not set.\n"
                "  Set your pre-deployed endpoint ID:\n"
                "  export SIGILANT_RUNPOD_ENDPOINT_ID=<your-endpoint-id>"
            )

        endpoint = runpod.Endpoint(endpoint_id)
        payload  = {"configs": [_config_to_dict(c) for c in configs]}

        job = endpoint.run(payload)
        results_raw = self._wait_for_job(job)

        return [_dict_to_result(c, d) for c, d in zip(configs, results_raw)]

    def _wait_for_job(self, job) -> list:
        deadline = time.time() + _JOB_TIMEOUT
        while time.time() < deadline:
            status = job.status()
            if status == "COMPLETED":
                return job.output().get("results", [])
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(f"RunPod job ended with status: {status}")
            time.sleep(_POLL_INTERVAL)
        raise TimeoutError("RunPod job did not complete within the timeout window.")


def _config_to_dict(c: RunConfig) -> dict:
    return {
        "quant_label":    c.quant_label,
        "context":        c.context,
        "batch":          c.batch,
        "kv_type":        c.kv_type,
        "model_repo":     c.model_repo,
        "model_filename": c.model_filename,
    }


def _dict_to_result(config: RunConfig, d: dict) -> RunResult:
    if d.get("error"):
        return RunResult(config=config, error=d["error"])
    return RunResult(
        config=config,
        tps=d.get("tps", 0.0),
        tps_p95=d.get("tps_p95"),
        ttft_ms=d.get("ttft_ms", 0.0),
        ttft_p95_ms=d.get("ttft_p95_ms"),
        itl_ms=d.get("itl_ms", 0.0),
        ppl=d.get("ppl"),
    )
