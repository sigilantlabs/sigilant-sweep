from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class RunConfig:
    quant_label: str        # e.g. "Q4_K_M"
    context: int            # n_ctx
    batch: int              # n_batch
    kv_type: str            # "k16v16" | "k8v8"
    regime: str = "default" # "default" | "long" | "depth_profile"
    depth_label: str = ""   # depth bucket for depth_profile regime ("8k"|"14k"|"28k")
    model_path: str = ""    # local GGUF path (local backend)
    model_repo: str = ""    # HF repo ID (remote backends)
    model_filename: str = ""  # filename within HF repo (remote backends)

    def label(self) -> str:
        base = f"{self.quant_label} · ctx:{self.context} · kv:{self.kv_type} · {self.regime}"
        if self.depth_label:
            return f"{base} · depth:{self.depth_label}"
        return base


@dataclass
class RunResult:
    config: RunConfig
    tps: float = 0.0
    tps_p95: Optional[float] = None
    ttft_ms: float = 0.0
    ttft_p95_ms: Optional[float] = None
    itl_ms: float = 0.0
    ppl: Optional[float] = None
    score: Optional[float] = None
    tps_norm: Optional[float] = None
    ttft_norm: Optional[float] = None
    ppl_norm: Optional[float] = None
    error: Optional[str] = None
    preflight: Optional[Dict[str, Any]] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None
