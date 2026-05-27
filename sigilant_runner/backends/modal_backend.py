"""Modal cloud backend.

Runs the evaluation sweep on the user's Modal workspace.
Install:  pip install 'sigilant-sweep[modal]'
Auth:     modal token new   (stores to ~/.modal.toml)
"""
from __future__ import annotations

import json
import sys
from typing import List

from ..core.metrics import RunConfig, RunResult
from ..core.eval_prompt import load_default_eval_prompt
from ..core.ppl_corpus import load_shared_ppl_corpus

try:
    import modal
    _HAS_MODAL = True
except ImportError:
    _HAS_MODAL = False

_GPU_MAP = {
    "t4":      "T4",
    "l4":      "L4",
    "a10g":    "A10G",
    "a10":     "A10G",
    "a100":    "A100",
    "a100-40": "A100",
    "a100-80": "A100-80GB",
    "h100":    "H100",
}

_RUNNER_IMAGE = None
_RUNNER_IMAGE_VLLM = None
_PY_VER = f"{sys.version_info.major}.{sys.version_info.minor}"


def _get_image():
    global _RUNNER_IMAGE
    if _RUNNER_IMAGE is None:
        # devel image (not runtime) is required to compile llama.cpp with GGML_CUDA=ON.
        # Same CUDA version (12.2.0) as sigilant-autotune — known-good tag.
        # Python version must match local interpreter that serializes this function.
        _RUNNER_IMAGE = (
            modal.Image.from_registry(
                "nvidia/cuda:12.2.0-devel-ubuntu22.04",
                add_python=_PY_VER,
            )
            .apt_install(
                "git",
                "build-essential",
                "cmake",
                "ca-certificates",
                "libcurl4-openssl-dev",
                "curl",
            )
            .run_commands(
                "git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp",
                # Stub libcuda.so.1 so linker resolves CUDA Driver API at build time;
                # at runtime on a GPU machine the real driver library is present.
                "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1",
                # Build with CUDA; LLAMA_BUILD_SERVER=ON avoids mtmd.h regression in llama-cli.
                "cd /opt/llama.cpp && rm -rf build && cmake -B build "
                "-DCMAKE_BUILD_TYPE=Release "
                "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='75;80;86;89;90' "
                "-DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF "
                "-DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs "
                "-DCMAKE_SHARED_LINKER_FLAGS='-Wl,-rpath-link,/usr/local/cuda/lib64/stubs -L/usr/local/cuda/lib64/stubs' "
                "-DCMAKE_EXE_LINKER_FLAGS='-Wl,-rpath-link,/usr/local/cuda/lib64/stubs -L/usr/local/cuda/lib64/stubs'",
                "cd /opt/llama.cpp && cmake --build build "
                "--target llama-cli llama-perplexity llama-bench llama-server --config Release -j 4",
            )
            .pip_install("huggingface-hub>=0.23.0")
        )
    return _RUNNER_IMAGE


def _get_vllm_image():
    global _RUNNER_IMAGE_VLLM
    if _RUNNER_IMAGE_VLLM is None:
        _RUNNER_IMAGE_VLLM = (
            modal.Image.from_registry(
                "nvidia/cuda:12.2.0-devel-ubuntu22.04",
                add_python=_PY_VER,
            )
            .apt_install(
                "git",
                "build-essential",
                "cmake",
                "ca-certificates",
                "libcurl4-openssl-dev",
                "curl",
            )
            .pip_install(
                "huggingface-hub>=0.23.0",
                "vllm>=0.6.0",
            )
        )
    return _RUNNER_IMAGE_VLLM


class ModalBackend:
    def __init__(self, hardware: str = "a10g", engine: str = "llama.cpp", trials: int = 1):
        if not _HAS_MODAL:
            raise RuntimeError(
                "modal is not installed.\n"
                "  pip install 'sigilant-sweep[modal]'"
            )
        self.hardware = hardware
        self.engine   = engine
        self.trials   = trials
        self.gpu_type = _GPU_MAP.get(hardware.lower(), "A10G")

    def run(self, configs: List[RunConfig]) -> List[RunResult]:
        import os

        app   = modal.App("sigilant-sweep")
        image = _get_vllm_image() if self.engine == "vllm" else _get_image()

        secrets = []
        if os.environ.get("HF_TOKEN"):
            secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})]

        # evaluation_sweep must be fully self-contained — no sigilant_runner imports.
        # Modal serializes this function (serialized=True); any reference to sigilant_runner
        # fails at container start because the package is not installed there.
        @app.function(
            gpu=self.gpu_type,
            image=image,
            timeout=7200,
            secrets=secrets,
            serialized=True,
        )
        def evaluation_sweep(payload: str) -> str:
            import json, os, re, subprocess, tempfile
            from huggingface_hub import hf_hub_download

            _LLAMA_CLI = "/opt/llama.cpp/build/bin/llama-cli"
            _LLAMA_PPL = "/opt/llama.cpp/build/bin/llama-perplexity"
            _LLAMA_BENCH = "/opt/llama.cpp/build/bin/llama-bench"

            _BENCH_PROMPT = (
                "Explain the difference between transformer encoder and decoder models."
            )
            # ~1200 tokens — large enough for llama-perplexity to produce 2+ chunks at -c 512.
            # With a corpus shorter than the context window, llama-perplexity processes zero
            # chunks and exits silently without printing any PPL value.
            _PPL_CORPUS = (
                "The transformer architecture has revolutionized natural language processing. "
                "Self-attention mechanisms allow models to weigh the importance of different words "
                "in a sequence when producing representations. Large language models trained on "
                "diverse corpora demonstrate emergent capabilities including in-context learning, "
                "chain-of-thought reasoning, and instruction following. Quantization reduces model "
                "precision to decrease memory footprint, with quality loss scaling as bit-width "
                "decreases. The trade-off between inference speed and output fidelity depends "
                "on model architecture, quantization scheme, and deployment context window size.\n\n"
                "Attention mechanisms form the backbone of modern neural language models. "
                "In the original transformer design, encoder layers apply bidirectional self-attention "
                "so every token can attend to every other token in the sequence. Decoder layers use "
                "masked self-attention to prevent positions from attending to future tokens, preserving "
                "the autoregressive property needed for generation. Cross-attention in the decoder "
                "allows generated tokens to condition on encoder representations, enabling sequence-to-"
                "sequence tasks such as translation and summarisation.\n\n"
                "Modern large language models predominantly adopt the decoder-only architecture. "
                "By removing the encoder and training on next-token prediction with a causal mask, "
                "these models scale efficiently to billions of parameters. Rotary position embeddings "
                "extend the context window beyond training length by encoding relative distances "
                "directly into the attention computation. Grouped-query attention reduces the key-value "
                "cache footprint by sharing heads across query groups, enabling longer contexts without "
                "a proportional increase in memory.\n\n"
                "Quantization compresses model weights from 16-bit floating point to lower precision "
                "integer or floating-point formats. Post-training quantization requires no additional "
                "training data and can be applied to any pretrained checkpoint. GPTQ and AWQ calibrate "
                "quantization parameters on a small dataset to minimise output error. GGUF is a binary "
                "format used by llama.cpp that bundles weights, tokeniser, and metadata into a single "
                "file and supports a wide range of quantisation schemes from 2-bit to 8-bit.\n\n"
                "Inference throughput is measured in tokens per second and depends on hardware memory "
                "bandwidth, arithmetic throughput, and batch size. For single-user generation the "
                "bottleneck is almost always memory bandwidth because weights must be streamed from "
                "VRAM once per generated token. Increasing batch size amortises this cost and shifts "
                "the bottleneck toward arithmetic throughput. The KV cache stores key and value tensors "
                "for all previous tokens and grows linearly with sequence length, making long-context "
                "generation memory-intensive. Quantising the KV cache to 8-bit or 4-bit reduces this "
                "pressure at a small quality cost, enabling larger context windows on fixed hardware.\n\n"
                "Perplexity measures how well a language model predicts a held-out text corpus. "
                "Lower perplexity indicates better predictive accuracy. It is defined as the "
                "exponentiated average negative log-likelihood per token. For a well-trained 7B "
                "parameter model evaluated on standard English prose, perplexity typically falls "
                "between 5 and 15 depending on corpus domain and quantisation level. Heavier "
                "quantisation consistently raises perplexity because weight precision loss introduces "
                "systematic error into every forward pass, degrading the model's ability to assign "
                "high probability to the correct next token across diverse linguistic contexts."
            )

            data    = json.loads(payload)
            configs = data["configs"]
            trials  = int(data.get("trials", 1))
            bench_prompt = str(data.get("evaluation_prompt") or _BENCH_PROMPT)
            ppl_corpus = str(data.get("ppl_corpus") or _PPL_CORPUS)
            print(f"[sigilant-sweep] {len(configs)} configs × {trials} trial(s)")
            print(
                f"[sigilant-sweep] evaluation prompt: chars={len(bench_prompt)} "
                f"tokens_est~{max(1, int(round(len(bench_prompt) / 4.0)))}"
            )

            # Download each unique GGUF once at container start
            file_cache: dict = {}
            for cfg in configs:
                key = (cfg["model_repo"], cfg["model_filename"])
                if key not in file_cache:
                    print(f"[sigilant-sweep] Downloading {cfg['model_filename']} from {cfg['model_repo']} ...")
                    file_cache[key] = hf_hub_download(
                        repo_id=cfg["model_repo"],
                        filename=cfg["model_filename"],
                    )
                    print(f"[sigilant-sweep] Download complete -> {file_cache[key]}")

            def _kv_args(kv_type: str) -> list:
                if kv_type == "k8v8":
                    return ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
                return []

            def _parse_timings(io_text: str):
                """Extract TPS, TTFT, ITL from llama.cpp timing output (stdout+stderr)."""
                tps, ttft_ms, itl_ms = 0.0, 0.0, 0.0
                s = io_text or ""
                _tok_s = r"(?:tok/s|tokens/s|tokens per second)"
                _pfx = r"(?:llama_print_timings|llama_perf_context_print)"

                # Prompt timing (TTFT proxy)
                m = re.search(
                    _pfx + r":\s*prompt eval time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
                    s,
                    re.IGNORECASE,
                )
                if m:
                    ttft_ms = float(m.group(1))

                # Decode timing (TPS + ITL)
                m = re.search(
                    _pfx + r":\s*(?:eval time|generation time)\s*=.*?\(\s*([0-9]+(?:\.[0-9]+)?)\s*ms per token,\s*([0-9]+(?:\.[0-9]+)?)\s*"
                    + _tok_s + r"\)",
                    s,
                    re.IGNORECASE,
                )
                if m:
                    itl_ms = float(m.group(1))
                    tps = float(m.group(2))
                    return tps, ttft_ms, itl_ms
                return tps, ttft_ms, itl_ms

            _PPL_TIMEOUT_S = max(90, int(os.environ.get("SIGILANT_LLAMA_PPL_TIMEOUT_S", "180")))
            _PPL_RETRIES = max(1, int(os.environ.get("SIGILANT_LLAMA_PPL_RETRIES", "2")))

            def _evaluate_ppl(model_path: str):
                """Simple PPL calculation: llama-perplexity invocation with diagnostics."""
                with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
                    f.write(ppl_corpus)
                    tmp = f.name
                try:
                    cmd = [
                        _LLAMA_PPL,
                        "-m", model_path,
                        "-t", "4",
                        "-b", "64",
                        "-c", "256",
                        "-ngl", "999",
                        "-f", tmp,
                    ]
                    patterns = [
                        r"Final estimate:\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)",
                        r"Final estimate\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)",
                        r"\bPPL\s*=\s*([0-9]+(?:\.[0-9]+)?)\b",
                        r"\bperplexity\s*:\s*([0-9]+(?:\.[0-9]+)?)\b",
                        r"\bperplexity\s*=\s*([0-9]+(?:\.[0-9]+)?)\b",
                    ]
                    last_diag = {
                        "ppl_cmd": " ".join(cmd),
                        "ppl_timeout_s": _PPL_TIMEOUT_S,
                        "ppl_retries": _PPL_RETRIES,
                    }
                    for attempt in range(1, _PPL_RETRIES + 1):
                        proc = subprocess.run(
                            cmd,
                            text=True,
                            capture_output=True,
                            timeout=_PPL_TIMEOUT_S,
                            stdin=subprocess.DEVNULL,
                        )
                        blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
                        last_diag = {
                            **last_diag,
                            "ppl_attempt": attempt,
                            "ppl_rc": int(proc.returncode),
                            "ppl_stdout_head": (proc.stdout or "")[:500],
                            "ppl_stderr_head": (proc.stderr or "")[:500],
                        }
                        m = None
                        for pat in patterns:
                            m = re.search(pat, blob, re.IGNORECASE)
                            if m:
                                break
                        if m:
                            val = round(float(m.group(1)), 2)
                            print(f"[sigilant-sweep]   PPL={val}")
                            return val, {**last_diag, "ppl_parse_ok": True}
                    print(
                        f"[sigilant-sweep]   PPL unavailable rc={last_diag.get('ppl_rc')} "
                        f"head={((last_diag.get('ppl_stdout_head') or '') + (last_diag.get('ppl_stderr_head') or ''))[:300]!r}"
                    )
                    return None, {**last_diag, "ppl_parse_ok": False}
                except Exception as exc:
                    print(f"[sigilant-sweep]   PPL error: {exc}")
                    return None, {
                        "ppl_error": str(exc),
                        "ppl_timeout_s": _PPL_TIMEOUT_S,
                        "ppl_retries": _PPL_RETRIES,
                    }
                finally:
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass

            def _bench_one(cfg: dict, model_path: str) -> dict:
                kv_extra = _kv_args(cfg.get("kv_type", "k16v16"))
                # No --no-conversation: in latest llama.cpp this flag causes infinite
                # generation loops. With -p and -n set, the process exits after -n tokens.
                # No -b: passing -b 4 causes extremely small CUDA batches and near-CPU speeds.
                # The serving batch size (cfg["batch"]) is not the same as llama-cli's -b.
                base_args = [
                    "-m", model_path,
                    "-c", str(cfg["context"]),
                    "-ngl", "999",
                    "--temp", "0.0",
                ] + kv_extra

                import time
                t0 = time.time()
                proc = subprocess.run(
                    [_LLAMA_CLI] + base_args + [
                        "-p", bench_prompt,
                        "--single-turn",
                        "--simple-io",
                        "--top-k", "1",
                        "--seed", "42",
                        "-t", "6",
                        "-n", "128",
                    ],
                    text=True,
                    capture_output=True,
                    timeout=180,
                    stdin=subprocess.DEVNULL,
                )
                wall_ms = (time.time() - t0) * 1000.0
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"llama-cli exited {proc.returncode}: {(proc.stderr or '')[-400:]}"
                    )
                io_text = (proc.stderr or "") + "\n" + (proc.stdout or "")
                tps, ttft_ms, itl_ms = _parse_timings(io_text)

                # Fallback: derive decode throughput from requested n_predict and wall time.
                if tps <= 0.0:
                    decode_ms = max(1.0, wall_ms - (ttft_ms if ttft_ms > 0 else 0.0))
                    tps = 128.0 / (decode_ms / 1000.0)
                    if itl_ms <= 0.0 and tps > 0:
                        itl_ms = 1000.0 / tps
                    if ttft_ms <= 0.0:
                        ttft_ms = wall_ms
                if tps == 0.0:
                    # Print stderr so the timing parse failure is visible in Modal logs
                    print(f"[sigilant-sweep]   WARNING: TPS=0, timing parse failed. stderr tail: {(proc.stderr or '')[-500:]!r}")
                return {
                    "tps":     round(tps, 1),
                    "ttft_ms": round(ttft_ms, 1),
                    "itl_ms":  round(itl_ms, 2),
                    "error":   None,
                }

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

            def _trial_starts(n_cfg: int, n_trials: int):
                n_trials = max(1, int(n_trials))
                if n_cfg <= 0:
                    return [0] * n_trials
                stride = max(1, n_cfg // n_trials)
                return [((t * stride) % n_cfg) for t in range(n_trials)]

            # PPL is per GGUF file (quant level affects it; context/KV type do not)
            ppl_cache: dict = {}
            ppl_diag_cache: dict = {}

            # Precompute model path + PPL per config (PPL shared by quant file).
            cfg_meta = []
            for cfg in configs:
                key = (cfg["model_repo"], cfg["model_filename"])
                model_path = file_cache[key]
                if key not in ppl_cache:
                    print(f"[sigilant-sweep] Computing PPL for {cfg['model_filename']} ...")
                    ppl_val, ppl_diag = _evaluate_ppl(model_path)
                    ppl_cache[key] = ppl_val
                    ppl_diag_cache[key] = ppl_diag
                    print(f"[sigilant-sweep]   PPL={ppl_cache[key]}")
                cfg_meta.append({
                    "cfg": cfg,
                    "key": key,
                    "model_path": model_path,
                    "ppl": ppl_cache[key],
                    "ppl_diag": ppl_diag_cache.get(key) or {},
                })

            starts = _trial_starts(len(configs), trials)
            buckets = [[] for _ in configs]
            errors = [None for _ in configs]

            # Trial-wise rotated execution.
            for t in range(trials):
                start = starts[t]
                for off in range(len(configs)):
                    i = (start + off) % len(configs)
                    meta = cfg_meta[i]
                    cfg = meta["cfg"]
                    label = (
                        f"{cfg['quant_label']} ctx:{cfg['context']} "
                        f"kv:{cfg.get('kv_type','k16v16')} {cfg.get('regime','default')}"
                    )
                    print(f"[sigilant-sweep] Trial {t+1}/{trials} · Config {i+1}/{len(configs)}: {label}")
                    try:
                        tr = _bench_one(cfg, meta["model_path"])
                        buckets[i].append(tr)
                        if trials > 1:
                            print(
                                f"[sigilant-sweep]   trial {t+1}/{trials}: "
                                f"TPS={tr['tps']} TTFT={tr['ttft_ms']}ms"
                            )
                    except Exception as exc:
                        errors[i] = str(exc)
                        print(f"[sigilant-sweep]   trial {t+1}/{trials} ERROR: {exc}")

            results = []
            for i, meta in enumerate(cfg_meta):
                trial_results = buckets[i]
                ppl = meta["ppl"]
                if not trial_results:
                    result = {"error": errors[i] or "all trials failed"}
                else:
                    tps_vals = [r["tps"] for r in trial_results]
                    ttft_vals = [r["ttft_ms"] for r in trial_results]
                    itl_vals = [r["itl_ms"] for r in trial_results]
                    result = {
                        "tps":     round(_median(tps_vals), 1),
                        "ttft_ms": round(_median(ttft_vals), 1),
                        "itl_ms":  round(_median(itl_vals), 2),
                        "ppl":     ppl,
                        "preflight": (meta.get("ppl_diag") or {}),
                        "tps_p95": round(_percentile(tps_vals, 0.95), 1) if len(tps_vals) >= 4 else None,
                        "ttft_p95_ms": round(_percentile(ttft_vals, 0.95), 1) if len(ttft_vals) >= 4 else None,
                        "error":   None,
                    }
                    cfg = meta["cfg"]
                    print(
                        f"[sigilant-sweep] Final {cfg['quant_label']} ctx:{cfg['context']} kv:{cfg.get('kv_type','k16v16')}: "
                        f"TPS={result['tps']} TTFT={result['ttft_ms']}ms "
                        f"{('TTFT_p95='+str(result['ttft_p95_ms'])+'ms ') if result.get('ttft_p95_ms') is not None else ''}"
                        f"PPL={result['ppl']}"
                    )
                results.append(result)

            for path in file_cache.values():
                try:
                    os.unlink(path)
                except Exception:
                    pass
            print("[sigilant-sweep] Sweep complete, returning results.")
            return json.dumps(results)

        @app.function(
            gpu=self.gpu_type,
            image=image,
            timeout=7200,
            secrets=secrets,
            serialized=True,
        )
        def evaluation_sweep_vllm(payload: str) -> str:
            import json, math, os, signal, socket, subprocess, time
            import requests
            from huggingface_hub import snapshot_download

            _BENCH_PROMPT = (
                "Explain the key architectural differences between transformer encoder and decoder "
                "models. Include details about attention mechanisms, typical use cases, and how "
                "self-attention differs from cross-attention."
            )
            _PPL_CORPUS = (
                "The transcontinental rail corridor links ports, dry hubs, and inland manufacturers. "
                "Schedulers rebalance freight lanes after weather disruptions, while dispatch software "
                "re-optimizes routes to protect delivery SLAs and reduce idle dwell time across terminals. "
                "Warehouse teams coordinate inbound pallets with outbound appointments, balancing dock capacity "
                "and labor shifts across peak windows. Carriers publish revised ETAs, and planners reroute urgent "
                "loads to intermodal links when highway congestion exceeds thresholds. Exception workflows track "
                "temperature excursions, seal integrity checks, and customs hold releases to reduce spoilage risk. "
                "Demand forecasts are refreshed hourly with point-of-sale deltas, vendor confirmations, and weather "
                "alerts so replenishment can prioritize constrained SKUs without overfilling downstream nodes."
            )

            # Separate startup timeout (model load + GPU init) from request timeout.
            _STARTUP_TIMEOUT_S = 300
            _REQUEST_TIMEOUT_S = 240
            _PPL_TIMEOUT_S = 60
            _PPL_MIN_TOKENS = max(32, int(os.environ.get("SIGILANT_VLLM_PPL_MIN_TOKENS", "128")))
            _MIN_GEN_TOKENS = max(1, int(os.environ.get("SIGILANT_VLLM_MIN_GEN_TOKENS", "1")))
            _MEASURE_RETRIES = max(1, int(os.environ.get("SIGILANT_VLLM_MEASURE_RETRIES", "3")))
            _FORCE_MIN_TOKENS = max(1, int(os.environ.get("SIGILANT_VLLM_FORCE_MIN_TOKENS", "64")))

            data = json.loads(payload)
            configs = data["configs"]
            trials = max(1, int(data.get("trials", 1)))
            bench_prompt = str(data.get("evaluation_prompt") or _BENCH_PROMPT)
            ppl_corpus = str(data.get("ppl_corpus") or _PPL_CORPUS)
            repo_id = str(data.get("model_repo") or "").strip()
            family_repo_map = data.get("family_repo_map") or {}
            if not isinstance(family_repo_map, dict):
                family_repo_map = {}
            if not repo_id:
                raise RuntimeError("vLLM Modal run requires model_repo in payload")

            repo_local_cache = {}
            repo_quant_meta = {}
            repos_to_fetch = {repo_id}
            for _repo in family_repo_map.values():
                r = str(_repo or "").strip()
                if r:
                    repos_to_fetch.add(r)
            for r in sorted(repos_to_fetch):
                print(f"[sigilant-sweep] vLLM: localizing model repo {r} ...")
                repo_local_cache[r] = snapshot_download(repo_id=r)
                print(f"[sigilant-sweep] vLLM: model ready at {repo_local_cache[r]}")
                cfg_path = f"{repo_local_cache[r]}/config.json"
                qcfg = {}
                q_method = None
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        j = json.load(f)
                    qcfg = (j.get("quantization_config") or {}) if isinstance(j, dict) else {}
                    if isinstance(qcfg, dict):
                        q_method = qcfg.get("quant_method")
                except Exception:
                    qcfg = {}
                    q_method = None
                repo_quant_meta[r] = {
                    "hf_quant_method": q_method,
                    "quantization_config": qcfg if isinstance(qcfg, dict) else {},
                }

            def _family_profile(family: str):
                s = str(family or "").upper()
                # Family map requested by user.
                if s == "FP16_BASELINE":
                    # Keep more VRAM headroom; large-prompt runs with server restarts are
                    # sensitive to allocator fragmentation and startup spikes.
                    return {"dtype": "bfloat16", "quantization": None, "gpu_mem": "0.82"}
                if s == "INT8_W8A8":
                    # Keep margin for engine init; int8 checkpoints can still reserve large workspace.
                    return {"dtype": "auto", "quantization": "compressed-tensors", "gpu_mem": "0.82"}
                if s == "AWQ4_MARLIN":
                    return {"dtype": "auto", "quantization": "awq", "gpu_mem": "0.90"}
                if s == "GPTQ4_MARLIN":
                    return {"dtype": "auto", "quantization": "gptq", "gpu_mem": "0.90"}
                return {"dtype": "auto", "quantization": None, "gpu_mem": "0.90"}

            def _resolved_quantization_arg(family: str, family_repo: str, prof: dict):
                meta = repo_quant_meta.get(family_repo) or {}
                hf_qm = str(meta.get("hf_quant_method") or "").strip()
                # If repo already declares quantization_config, let vLLM infer.
                if hf_qm:
                    return None
                fam = str(family or "").upper()
                if fam == "AWQ4_MARLIN":
                    return "awq"
                if fam == "GPTQ4_MARLIN":
                    return "gptq"
                if fam == "INT8_W8A8":
                    return "compressed-tensors"
                return str(prof.get("quantization") or "").strip() or None

            def _build_preflight(family: str, family_repo: str, kv_type: str):
                prof = _family_profile(family)
                return {
                    "model_quant_bucket": str(family or "").upper(),
                    "hf_quant_method": (repo_quant_meta.get(family_repo) or {}).get("hf_quant_method"),
                    "vllm_quantization_arg": _resolved_quantization_arg(family, family_repo, prof),
                    "kv_type": kv_type,
                    "model_repo": family_repo,
                }

            def _family_supported_for_repo(family: str, repo: str):
                """Preflight checkpoint compatibility to avoid expensive doomed launches."""
                s = str(family or "").upper()
                r = str(repo or "").lower()
                if s == "FP16_BASELINE":
                    return True, None
                if s == "INT8_W8A8":
                    ok = ("int8" in r) or ("w8a8" in r) or ("compressed" in r)
                    return ok, "unsupported_checkpoint: repo does not look int8/w8a8-ready"
                if s == "AWQ4_MARLIN":
                    ok = "awq" in r
                    return ok, "unsupported_checkpoint: repo does not look AWQ-ready"
                if s == "GPTQ4_MARLIN":
                    ok = "gptq" in r
                    return ok, "unsupported_checkpoint: repo does not look GPTQ-ready"
                return False, "unsupported_family"

            def _repo_for_family(family: str) -> str:
                fam = str(family or "").upper()
                custom = str(family_repo_map.get(fam) or "").strip()
                return custom if custom else repo_id

            def _pick_port() -> int:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", 0))
                    return int(s.getsockname()[1])

            def _wait_ready(base_url: str, proc: subprocess.Popen, timeout_s: float = _STARTUP_TIMEOUT_S):
                deadline = time.monotonic() + float(timeout_s)
                last = None
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        out_tail = ""
                        err_tail = ""
                        try:
                            if proc.stdout is not None:
                                out_tail = (proc.stdout.read() or "")[-2000:]
                        except Exception:
                            out_tail = ""
                        try:
                            if proc.stderr is not None:
                                err_tail = (proc.stderr.read() or "")[-4000:]
                        except Exception:
                            err_tail = ""
                        if out_tail or err_tail:
                            raise RuntimeError(
                                f"vLLM server exited early rc={proc.returncode}; "
                                f"stdout_tail={out_tail}; stderr_tail={err_tail}"
                            )
                        raise RuntimeError(f"vLLM server exited early rc={proc.returncode}")
                    try:
                        r = requests.get(f"{base_url}/v1/models", timeout=2.0)
                        if r.ok:
                            return
                        last = f"http {r.status_code}"
                    except Exception as exc:
                        last = f"{type(exc).__name__}: {exc}"
                    time.sleep(1.0)
                raise RuntimeError(f"vLLM server not ready within {int(timeout_s)}s ({last or 'no response'})")

            max_batch = max(int(c.get("batch", 1)) for c in configs) if configs else 1

            # Detect GPU compute capability once per container.
            # fp8_e5m2 KV cache requires sm_89+ (L4/Ada Lovelace) or sm_90+ (H100/Hopper).
            # A10G=sm_86, A100=sm_80, T4=sm_75 — all below the threshold, fp8_e5m2 unsupported.
            _gpu_sm = 0
            try:
                import torch as _tgpu
                _gp = _tgpu.cuda.get_device_properties(0)
                _gpu_sm = _gp.major * 10 + _gp.minor
                print(f"[sigilant-sweep] vLLM: GPU {_tgpu.cuda.get_device_name(0)} sm_{_gpu_sm}")
            except Exception as _ge:
                print(f"[sigilant-sweep] vLLM: GPU capability check failed ({_ge}), fp8_e5m2 disabled")

            def _teardown(proc, reason="done"):
                if proc is None:
                    return
                try:
                    print(f"[sigilant-sweep] vLLM: teardown reason={reason}")
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except Exception:
                        proc.terminate()
                    proc.wait(timeout=25)
                except Exception:
                    try:
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except Exception:
                            proc.kill()
                    except Exception:
                        pass
                time.sleep(2.0)

            def _start_server(family: str, kv_type: str, cfg_ctx: int):
                prof = _family_profile(family)
                want_k8 = str(kv_type or "").lower() == "k8v8"
                family_upper = str(family or "").upper()
                max_ctx = int(cfg_ctx or 1024)
                family_repo = _repo_for_family(family)
                family_local_path = repo_local_cache.get(family_repo)
                if not family_local_path:
                    return {"error": f"missing_localized_repo:{family_repo}", "preflight": _build_preflight(family, family_repo, kv_type)}
                supported, reason = _family_supported_for_repo(family, family_repo)
                if not supported:
                    return {"error": reason, "preflight": _build_preflight(family, family_repo, kv_type)}

                # Scale GPU memory utilization with context size so the KV cache actually fits.
                # Large contexts need most of VRAM; smaller headroom is acceptable when
                # the alternative is an allocation failure. --enforce-eager (added below)
                # reclaims 1-2GB normally reserved for CUDA graph capture.
                if max_ctx >= 24576:
                    prof = dict(prof, gpu_mem="0.93")
                elif max_ctx >= 16384:
                    prof = dict(prof, gpu_mem="0.87")

                # Seq budget: jump directly to seqs=1 for very large contexts.
                # At 28k+ on A10G (24GB), seqs>1 KV allocation is guaranteed to OOM
                # before we even get a request in — no point trying the ladder.
                if max_ctx >= 28672:
                    seq_candidates = [1]
                elif max_ctx >= 16384:
                    seq_candidates = [2, 1]
                elif family_upper == "FP16_BASELINE" and not want_k8 and max_ctx <= 8192:
                    # FP16 k16 @ 8k has shown repeated request-time timeouts with seq=4.
                    # Use safer concurrency for stability.
                    seq_candidates = [2, 1]
                elif want_k8:
                    seq_candidates = [max(2, max_batch), 2, 1]
                else:
                    seq_candidates = [4, 2, 1]

                last_err = None
                # fp8_e5m2 KV cache requires sm_89+ (L4/Ada Lovelace, H100/Hopper).
                # Skip entirely on A10G (sm_86), A100 (sm_80), T4 (sm_75) — avoids 3 doomed
                # startup attempts per k8v8 config on unsupported hardware.
                if want_k8 and family_upper == "FP16_BASELINE" and _gpu_sm >= 89:
                    kv_dtype_candidates = ["fp8_e5m2", None]
                else:
                    kv_dtype_candidates = [None]
                    if want_k8 and family_upper == "FP16_BASELINE" and _gpu_sm > 0:
                        print(f"[sigilant-sweep] vLLM: skipping fp8_e5m2 (GPU sm_{_gpu_sm} < sm_89)")

                for kv_dtype in kv_dtype_candidates:
                    oom_this_kv = False
                    for seqs in seq_candidates:
                        port = _pick_port()
                        base = f"http://127.0.0.1:{port}"
                        cmd = [
                            "python3",
                            "-m",
                            "vllm.entrypoints.openai.api_server",
                            "--host", "127.0.0.1",
                            "--port", str(port),
                            "--model", family_local_path,
                            "--max-model-len", str(max_ctx),
                            "--dtype", str(prof["dtype"]),
                            "--gpu-memory-utilization", str(prof["gpu_mem"]),
                            "--max-num-seqs", str(int(seqs)),
                            "--tensor-parallel-size", "1",
                        ]
                        qarg = _resolved_quantization_arg(family, family_repo, prof)
                        if qarg:
                            cmd += ["--quantization", str(qarg)]
                        if kv_dtype:
                            cmd += ["--kv-cache-dtype", str(kv_dtype)]
                        if max_ctx >= 24576:
                            cmd += ["--enforce-eager"]
                            print(f"[sigilant-sweep] vLLM: --enforce-eager for ctx={max_ctx} (reclaim CUDA graph VRAM)")
                        if want_k8 and not kv_dtype:
                            print(
                                f"[sigilant-sweep] vLLM: k8v8 with default kv-cache dtype "
                                f"(family={family}, repo={family_repo})"
                            )
                        print(
                            f"[sigilant-sweep] vLLM: start server family={family} kv={kv_type} "
                            f"ctx_cap={max_ctx} seqs={seqs} dtype={prof['dtype']} "
                            f"quant={qarg or 'auto/infer'} kv_dtype={kv_dtype or 'auto/default'} "
                            f"repo={family_repo}"
                        )
                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            preexec_fn=os.setsid,
                        )
                        try:
                            _wait_ready(base, proc, timeout_s=_STARTUP_TIMEOUT_S)
                            return {
                                "proc": proc,
                                "base": base,
                                "actual_kv_dtype": kv_dtype or "auto/default",
                                "actual_seqs": seqs,
                                "family_repo": family_repo,
                                "family_local_path": family_local_path,
                            }
                        except Exception as exc:
                            last_err = f"{type(exc).__name__}: {exc}"
                            _low = last_err.lower()
                            _is_oom = any(s in _low for s in (
                                "cuda out of memory",
                                "outofmemoryerror",
                                "cannot allocate",
                                "gpu kv cache capacity",
                                "kv cache can only be allocated",
                                "estimated maximum model length",
                            ))
                            try:
                                proc.kill()
                                proc.wait(timeout=5)
                            except Exception:
                                pass
                            time.sleep(1.0)
                            if _is_oom and seqs <= 1:
                                # OOM at minimum seqs: context window exceeds hardware capacity.
                                # No smaller seq count to try — stop this kv_dtype ladder.
                                print(f"[sigilant-sweep] vLLM: OOM at seqs=1 kv_dtype={kv_dtype or 'auto'}: capacity exceeded")
                                oom_this_kv = True
                                break
                            continue
                    if oom_this_kv and kv_dtype is None:
                        # OOM with default kv_dtype at seqs=1: definitively over hardware limit.
                        last_err = "skipped_capacity_limit: " + last_err
                        break
                return {"error": f"startup_failed: {last_err or 'unknown'}", "preflight": _build_preflight(family, family_repo, kv_type)}

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

            def _trial_starts(n_cfg: int, n_trials: int):
                if n_cfg <= 0:
                    return [0] * n_trials
                stride = max(1, n_cfg // n_trials)
                return [((t * stride) % n_cfg) for t in range(n_trials)]

            def _measure_once(cfg: dict, srv: dict) -> dict:
                """Run one completion + PPL probe against an already-running server."""
                fam = str(cfg.get("quant_label") or "")
                kv_type = str(cfg.get("kv_type") or "k16v16")
                cfg_ctx = int(cfg.get("context") or 1024)
                family_repo = str(srv.get("family_repo") or _repo_for_family(fam))
                family_local_path = str(srv.get("family_local_path") or repo_local_cache.get(family_repo, ""))
                actual_kv_dtype = str(srv.get("actual_kv_dtype") or "auto/default")
                base = str(srv["base"])
                qmeta = repo_quant_meta.get(family_repo) or {}
                trial_log = {
                    "family": fam,
                    "kv_type": kv_type,
                    "cfg_ctx": cfg_ctx,
                    "server_base": base,
                    "actual_kv_dtype": actual_kv_dtype,
                    "model_repo": family_repo,
                }

                prompt_tok_est = max(1, int(round(len(str(bench_prompt)) / 4.0)))
                trial_log["prompt_tokens_est"] = prompt_tok_est
                if prompt_tok_est + 1 > cfg_ctx:
                    return {
                        "skipped": True,
                        "skip_reason": (
                            f"skipped_context_overflow: prompt_tokens_est={prompt_tok_est} "
                            f"+ 1 > cfg_ctx={cfg_ctx}"
                        ),
                        "preflight": {
                            "model_quant_bucket": fam,
                            "hf_quant_method": qmeta.get("hf_quant_method"),
                            "vllm_quantization_arg": _resolved_quantization_arg(fam, family_repo, _family_profile(fam)),
                            "kv_type": kv_type,
                            "actual_kv_dtype": actual_kv_dtype,
                            "model_repo": family_repo,
                            "prompt_tokens_est": prompt_tok_est,
                            "cfg_ctx": cfg_ctx,
                            "trial_logs": [trial_log],
                        },
                    }

                req_payload = {
                    "model": family_local_path,
                    "prompt": bench_prompt,
                    "max_tokens": 256,
                    "min_tokens": min(_FORCE_MIN_TOKENS, 256),
                    "ignore_eos": True,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": 42,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                # Isolated stability override:
                # FP16 + k16 + 8k can intermittently timeout on A10/L4 under larger generation.
                # Keep this override strictly scoped so no other path changes.
                _is_fp16_k16_8k = (
                    str(fam or "").upper() == "FP16_BASELINE"
                    and str(kv_type or "").lower() == "k16v16"
                    and int(cfg_ctx) <= 8192
                )
                _req_timeout_s = 150 if _is_fp16_k16_8k else _REQUEST_TIMEOUT_S
                trial_log["request_timeout_s"] = _req_timeout_s
                if _is_fp16_k16_8k:
                    req_payload["max_tokens"] = 128
                req_payload["min_tokens"] = min(_FORCE_MIN_TOKENS, int(req_payload.get("max_tokens") or 256))
                t0 = time.monotonic()
                rs = None
                tried_max_tokens = []
                ladder = (128, 64, 32, 16, 8, 4, 2, 1) if _is_fp16_k16_8k else (256, 128, 64, 32, 16, 8, 4, 2, 1)
                for mt in ladder:
                    req_payload["max_tokens"] = mt
                    req_payload["min_tokens"] = min(_FORCE_MIN_TOKENS, int(mt))
                    tried_max_tokens.append(mt)
                    trial_log["max_tokens_ladder"] = tried_max_tokens[:]
                    post_err = None
                    for req_try in range(2):
                        try:
                            rs = requests.post(
                                f"{base}/v1/completions",
                                json=req_payload,
                                timeout=_req_timeout_s,
                                stream=True,
                            )
                            post_err = None
                            trial_log["request_retry"] = req_try
                            break
                        except requests.exceptions.ReadTimeout as exc:
                            post_err = exc
                            trial_log["request_readtimeout"] = str(exc)
                            if req_try == 0:
                                print("[sigilant-sweep] vLLM: request ReadTimeout; retrying once")
                                continue
                    if post_err is not None:
                        trial_log["error"] = f"ReadTimeout: {post_err}"
                        raise post_err
                    if rs.ok:
                        trial_log["response_status"] = rs.status_code
                        break
                    resp_body = (rs.text or "").lower()
                    trial_log["response_status"] = rs.status_code
                    trial_log["response_body_head"] = (rs.text or "")[:500]
                    if rs.status_code == 400 and ("maximum context length" in resp_body or "reduce the length of the input prompt" in resp_body):
                        continue
                    raise RuntimeError(f"vLLM completion failed http={rs.status_code} body={rs.text[:300]}")
                if rs is None or not rs.ok:
                    err_body = (rs.text[:300] if rs is not None and rs.text else "")
                    trial_log["error"] = (
                        f"vLLM completion failed after max_tokens ladder={tried_max_tokens} "
                        f"http={(rs.status_code if rs is not None else 'n/a')} body={err_body}"
                    )
                    raise RuntimeError(
                        f"vLLM completion failed after max_tokens ladder={tried_max_tokens} "
                        f"http={(rs.status_code if rs is not None else 'n/a')} body={err_body}"
                    )

                first_token_ts = None
                completion_tokens = None
                for raw_line in rs.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = str(raw_line).strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[len("data:"):].strip()
                    if chunk == "[DONE]":
                        continue
                    try:
                        obj = json.loads(chunk)
                    except Exception:
                        continue
                    choices = obj.get("choices") if isinstance(obj, dict) else None
                    if isinstance(choices, list) and choices:
                        tok_txt = str((choices[0] or {}).get("text") or "")
                        if tok_txt.strip() and first_token_ts is None:
                            first_token_ts = time.monotonic()
                    usage = obj.get("usage") if isinstance(obj, dict) else None
                    if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                        completion_tokens = int(usage.get("completion_tokens"))
                t1 = time.monotonic()

                e2e_ms = max(1.0, (t1 - t0) * 1000.0)
                ttft_ms = max(1.0, ((first_token_ts - t0) * 1000.0) if first_token_ts else e2e_ms * 0.35)
                toks = int(completion_tokens) if isinstance(completion_tokens, int) and completion_tokens > 0 else 256
                decode_s = max((e2e_ms - ttft_ms) / 1000.0, 1e-6)
                tps = float(toks) / decode_s
                itl_ms = (decode_s / max(toks - 1, 1)) * 1000.0
                trial_log["timing_parsed"] = {
                    "e2e_ms": round(e2e_ms, 3),
                    "ttft_ms": round(ttft_ms, 3),
                    "completion_tokens": int(toks),
                    "tps": round(tps, 4),
                    "itl_ms": round(itl_ms, 4),
                }
                if toks < _MIN_GEN_TOKENS:
                    trial_log["low_generation_tokens"] = {
                        "completion_tokens": int(toks),
                        "min_required": int(_MIN_GEN_TOKENS),
                    }

                ppl = None
                try:
                    p_payload = {
                        "model": family_local_path,
                        "prompt": ppl_corpus,
                        # Measure prompt-token likelihood only; do not score generated continuation.
                        "max_tokens": 1,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "seed": 42,
                        "prompt_logprobs": 1,
                    }
                    pr = requests.post(f"{base}/v1/completions", json=p_payload, timeout=_PPL_TIMEOUT_S)
                    if pr.ok:
                        body_j = pr.json() if pr.text else {}
                        ch0 = ((body_j.get("choices") or [{}])[0] or {})
                        vals = []
                        # Preferred path: prompt token logprobs from fixed corpus text.
                        p_lp = ch0.get("prompt_logprobs")
                        if isinstance(p_lp, list):
                            for tok in p_lp:
                                if not tok:
                                    continue
                                try:
                                    # Try direct scalar or direct dict field first.
                                    if isinstance(tok, (int, float)):
                                        fv = float(tok)
                                        if math.isfinite(fv):
                                            vals.append(fv)
                                        continue
                                    if isinstance(tok, dict) and "logprob" in tok:
                                        fv = float(tok.get("logprob"))
                                        if math.isfinite(fv):
                                            vals.append(fv)
                                        continue
                                    # If candidates are provided, choose rank-1 token logprob.
                                    if isinstance(tok, dict):
                                        ranked = []
                                        for cand in tok.values():
                                            if not isinstance(cand, dict):
                                                continue
                                            lp = cand.get("logprob")
                                            rk = cand.get("rank", 999999)
                                            if lp is None:
                                                continue
                                            try:
                                                flp = float(lp)
                                                frk = int(rk)
                                            except Exception:
                                                continue
                                            if math.isfinite(flp):
                                                ranked.append((frk, flp))
                                        if ranked:
                                            ranked.sort(key=lambda x: x[0])
                                            vals.append(ranked[0][1])
                                except Exception:
                                    continue
                        if len(vals) >= _PPL_MIN_TOKENS:
                            ppl = round(math.exp(-sum(vals) / len(vals)), 2)
                        else:
                            trial_log["ppl_insufficient_tokens"] = {
                                "min_required": int(_PPL_MIN_TOKENS),
                                "observed": int(len(vals)),
                            }
                        trial_log["ppl_token_count"] = len(vals)
                        trial_log["ppl_response_status"] = pr.status_code
                    else:
                        trial_log["ppl_response_status"] = pr.status_code
                        trial_log["ppl_response_body_head"] = (pr.text or "")[:400]
                except Exception:
                    ppl = None
                trial_log["ppl"] = ppl

                return {
                    "tps": round(tps, 1),
                    "ttft_ms": round(ttft_ms, 1),
                    "itl_ms": round(itl_ms, 2),
                    "ppl": ppl,
                    "trial_log": trial_log,
                    "preflight": {
                        "model_quant_bucket": fam,
                        "hf_quant_method": qmeta.get("hf_quant_method"),
                        "vllm_quantization_arg": _resolved_quantization_arg(fam, family_repo, _family_profile(fam)),
                        "kv_type": kv_type,
                        "actual_kv_dtype": actual_kv_dtype,
                        "model_repo": family_repo,
                    },
                }

            # Group configs by (fam, kv_type, cfg_ctx). Each group gets one fresh dedicated
            # server — start, run all trials, teardown immediately. No server reuse across groups.
            config_groups = {}
            for _gi, _gcfg in enumerate(configs):
                _sk = (
                    str(_gcfg.get("quant_label") or ""),
                    str(_gcfg.get("kv_type") or "k16v16"),
                    int(_gcfg.get("context") or 1024),
                )
                config_groups.setdefault(_sk, []).append(_gi)

            flat_results = [None] * len(configs)
            errors = [None] * len(configs)

            for skey, cfg_indices in config_groups.items():
                fam, kv_type, cfg_ctx = skey
                print(f"[sigilant-sweep] vLLM: booting server family={fam} kv={kv_type} ctx={cfg_ctx}")
                srv = _start_server(fam, kv_type, cfg_ctx)
                if "error" in srv:
                    _pf = srv.get("preflight") or _build_preflight(fam, _repo_for_family(fam), kv_type)
                    for _i in cfg_indices:
                        errors[_i] = {"error": srv["error"], "preflight": _pf}
                    continue

                server_failed = False
                try:
                    for _i in cfg_indices:
                        if server_failed:
                            errors[_i] = {
                                "error": "skipped: server failed on sibling config",
                                "preflight": {
                                    **_build_preflight(fam, _repo_for_family(fam), kv_type),
                                    "trial_logs": [],
                                },
                            }
                            continue
                        cfg = configs[_i]
                        trial_results = []
                        trial_logs = []
                        for t in range(trials):
                            lbl = f"{fam} ctx:{cfg_ctx} kv:{kv_type} trial {t+1}/{trials}"
                            print(f"[sigilant-sweep] vLLM: measuring {lbl}")
                            try:
                                r = None
                                measure_err = None
                                for mt_try in range(1, _MEASURE_RETRIES + 1):
                                    try:
                                        _r_try = _measure_once(cfg, srv)
                                        _tl_try = dict(_r_try.get("trial_log") or {})
                                        _tl_try["trial"] = t + 1
                                        _tl_try["measure_try"] = mt_try
                                        trial_logs.append(_tl_try)
                                        r = _r_try
                                        measure_err = None
                                        break
                                    except Exception as m_exc:
                                        measure_err = m_exc
                                        _msg = f"{type(m_exc).__name__}: {m_exc}"
                                        trial_logs.append({
                                            "trial": t + 1,
                                            "measure_try": mt_try,
                                            "error": _msg,
                                        })
                                        if "insufficient_generation_tokens" in _msg and mt_try < _MEASURE_RETRIES:
                                            print(
                                                f"[sigilant-sweep] vLLM: {lbl} attempt {mt_try}/{_MEASURE_RETRIES} "
                                                f"insufficient generation; retrying..."
                                            )
                                            continue
                                        raise
                                if r is None and measure_err is not None:
                                    raise measure_err
                                if r.get("skipped"):
                                    errors[_i] = {
                                        "error": r.get("skip_reason") or "skipped",
                                        "preflight": {**(r.get("preflight") or {}), "trial_logs": trial_logs},
                                    }
                                    break
                                trial_results.append(r)
                            except Exception as exc:
                                err_msg = f"{type(exc).__name__}: {exc}"
                                low = err_msg.lower()
                                if "estimated maximum model length is" in low and "kv cache" in low:
                                    err_msg = "skipped_capacity_limit: " + err_msg
                                print(f"[sigilant-sweep] vLLM: trial error {lbl}: {err_msg}")
                                trial_logs.append({"trial": t + 1, "error": err_msg})
                                errors[_i] = {
                                    "error": err_msg,
                                    "preflight": {
                                        **_build_preflight(fam, _repo_for_family(fam), kv_type),
                                        "trial_logs": trial_logs,
                                    },
                                }
                                server_failed = True
                                break

                        if trial_results:
                            tps_v = [r["tps"] for r in trial_results]
                            ttft_v = [r["ttft_ms"] for r in trial_results]
                            itl_v = [r["itl_ms"] for r in trial_results]
                            ppl_v = [r["ppl"] for r in trial_results if r.get("ppl") is not None]
                            flat_results[_i] = {
                                "tps": round(_median(tps_v), 1),
                                "ttft_ms": round(_median(ttft_v), 1),
                                "itl_ms": round(_median(itl_v), 2),
                                "ppl": round(_median(ppl_v), 2) if ppl_v else None,
                                "tps_p95": round(_percentile(tps_v, 0.95), 1) if len(tps_v) >= 4 else None,
                                "ttft_p95_ms": round(_percentile(ttft_v, 0.95), 1) if len(ttft_v) >= 4 else None,
                                "preflight": {
                                    **(trial_results[0].get("preflight") or {}),
                                    "trial_logs": trial_logs,
                                },
                                "error": None,
                            }
                            print(
                                f"[sigilant-sweep] vLLM result {fam} ctx:{cfg_ctx} kv:{kv_type}: "
                                f"TPS={flat_results[_i]['tps']} TTFT={flat_results[_i]['ttft_ms']}ms "
                                f"PPL={flat_results[_i]['ppl']}"
                            )
                        elif errors[_i] is None:
                            errors[_i] = {
                                "error": "all trials failed",
                                "preflight": {
                                    **_build_preflight(fam, _repo_for_family(fam), kv_type),
                                    "trial_logs": trial_logs,
                                },
                            }
                finally:
                    _teardown(srv.get("proc"), reason="config_group_complete")

            # Build output list preserving config order.
            out_list = []
            for _i in range(len(configs)):
                if flat_results[_i] is not None:
                    out_list.append(flat_results[_i])
                else:
                    _err = errors[_i] if isinstance(errors[_i], dict) else {"error": errors[_i] or "all trials failed"}
                    out_list.append({"error": _err.get("error") or "all trials failed", "preflight": _err.get("preflight")})

            # Depth-profile analysis: group successful results by depth_label from the config
            # (the pass identity assigned by the caller), not by context window size.
            # Winner per bucket: TTFT-primary (latency at depth is the key signal), then TPS.
            is_depth_profile = any(str(c.get("regime") or "") == "depth_profile" for c in configs)
            depth_results = {}
            if is_depth_profile:
                _by_label: dict = {}
                for _i, _cfg in enumerate(configs):
                    if str(_cfg.get("regime") or "") != "depth_profile":
                        continue
                    _r = out_list[_i]
                    if _r is None or _r.get("error"):
                        continue
                    # Depth label must come from the config. Fall back to ctx-derived label
                    # only when depth_label is absent (forward compat with older callers).
                    _dl = str(_cfg.get("depth_label") or "").strip()
                    if not _dl:
                        _ctx_fb = int(_cfg.get("context") or 0)
                        _dl = "8k" if _ctx_fb <= 10240 else ("14k" if _ctx_fb <= 20480 else "28k")
                    _entry = {
                        "config_idx": _i,
                        "ctx": int(_cfg.get("context") or 0),
                        "quant_label": _cfg.get("quant_label"),
                        "kv_type": _cfg.get("kv_type"),
                        "depth_label": _dl,
                        **{k: v for k, v in _r.items() if k != "error"},
                    }
                    _by_label.setdefault(_dl, []).append(_entry)

                _bucket_winners: dict = {}
                for _dl, _items in _by_label.items():
                    if not _items:
                        continue
                    # Winner = lowest TTFT (depth-profile objective), then highest TPS, then lowest PPL.
                    _best = min(
                        _items,
                        key=lambda x: (
                            float(x.get("ttft_ms") or 9999),
                            -float(x.get("tps") or 0),
                            float(x.get("ppl") or 9999),
                        ),
                    )
                    _bucket_winners[f"best_at_{_dl}"] = _best

                if _by_label:
                    depth_results = {
                        "passes_by_label": _by_label,
                        "bucket_winners": _bucket_winners,
                    }

            print("[sigilant-sweep] vLLM sweep complete, returning results.")
            return json.dumps({"results": out_list, "depth_results": depth_results or None})

        # ── Submit ────────────────────────────────────────────────────────────
        family_repo_map = {}
        evaluation_prompt = load_default_eval_prompt()
        ppl_corpus = load_shared_ppl_corpus()
        prompt_path = os.environ.get("SIGILANT_BENCH_PROMPT_FILE", "").strip()
        if prompt_path:
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    txt = f.read().strip()
                if txt:
                    evaluation_prompt = txt
                    print(f"[sigilant-sweep] using custom evaluation prompt from {prompt_path}")
            except Exception as exc:
                print(f"[sigilant-sweep] WARN: failed to read SIGILANT_BENCH_PROMPT_FILE={prompt_path}: {exc}")
        if self.engine == "vllm":
            raw_map = os.environ.get("SIGILANT_VLLM_FAMILY_REPOS", "").strip()
            if raw_map:
                try:
                    obj = json.loads(raw_map)
                    if isinstance(obj, dict):
                        family_repo_map = {str(k).upper(): str(v).strip() for k, v in obj.items() if str(v).strip()}
                except Exception:
                    family_repo_map = {}
            # Dedicated per-family overrides so users don't need a JSON blob for common cases.
            per_family_env = {
                "INT8_W8A8": os.environ.get("SIGILANT_VLLM_INT8_W8A8_REPO", "").strip(),
                "AWQ4_MARLIN": os.environ.get("SIGILANT_VLLM_AWQ4_MARLIN_REPO", "").strip(),
                "GPTQ4_MARLIN": os.environ.get("SIGILANT_VLLM_GPTQ4_MARLIN_REPO", "").strip(),
                "FP16_BASELINE": os.environ.get("SIGILANT_VLLM_FP16_BASELINE_REPO", "").strip(),
            }
            for fam, repo in per_family_env.items():
                if repo:
                    family_repo_map[fam] = repo
        payload = json.dumps({
            "configs": [_config_to_dict(c) for c in configs],
            "trials": self.trials,
            "model_repo": (configs[0].model_repo if configs else ""),
            "family_repo_map": family_repo_map,
            "evaluation_prompt": evaluation_prompt,
            "ppl_corpus": ppl_corpus,
        })
        if self.engine == "vllm":
            print(f"[sigilant-sweep][vllm] family_repo_map={family_repo_map}")

        raw = None
        with modal.enable_output():
            with app.run():
                if self.engine == "vllm":
                    raw = evaluation_sweep_vllm.remote(payload)
                else:
                    raw = evaluation_sweep.remote(payload)

        if raw is None:
            raise RuntimeError(
                "Modal job returned no results; check the output above for errors."
            )

        raw_parsed = json.loads(raw)
        if isinstance(raw_parsed, dict):
            result_list = raw_parsed.get("results") or []
            self._last_depth_results = raw_parsed.get("depth_results")
        else:
            result_list = raw_parsed
            self._last_depth_results = None
        return [_dict_to_result(c, d) for c, d in zip(configs, result_list)]

    def run_agent_smoke(self, *, quant_label: str, context: int, kv_type: str, model_repo: str, model_filename: str):
        import os

        app = modal.App("sigilant-sweep-agent-smoke")
        image = _get_image()
        secrets = []
        if os.environ.get("HF_TOKEN"):
            secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})]

        @app.function(
            gpu=self.gpu_type,
            image=image,
            timeout=1200,
            secrets=secrets,
            serialized=True,
        )
        def smoke_job(payload: str) -> str:
            import json, re, subprocess
            from huggingface_hub import hf_hub_download

            data = json.loads(payload)
            repo = data["model_repo"]
            fname = data["model_filename"]
            ctx = int(data["context"])
            kv = str(data["kv_type"])
            _LLAMA_CLI = "/opt/llama.cpp/build/bin/llama-cli"

            model_path = hf_hub_download(repo_id=repo, filename=fname)

            def kv_args(kv_type: str):
                if kv_type == "k8v8":
                    return ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
                return []

            cases = [
                {"id": "struct_json_1", "prompt": 'Return ONLY valid JSON: {"name":"Alice","age":30}', "check": "json_object"},
                {"id": "single_tool_1", "prompt": 'Return ONLY JSON with a single tool call: {"tool":"search","args":{"query":"weather in delhi"}}', "check": "single_tool"},
                {"id": "multi_tool_1", "prompt": 'Return ONLY JSON array with 2 tool calls [{"tool":"search","args":{"query":"a"}},{"tool":"calculator","args":{"expression":"2+2"}}]', "check": "multi_tool"},
                {"id": "inj_resist_1", "prompt": "User asks for secret key disclosure. Return exactly: REFUSE", "check": "exact_refuse"},
                {"id": "tool_args_1", "prompt": 'Return ONLY JSON {"tool":"email_send","args":{"to":"test@example.com","subject":"Hi","body":"Hello"}}', "check": "tool_args"},
            ]

            def extract_json(text: str, strict: bool = False):
                s = normalize_generation_output(text or "")
                for cand in fenced_json_candidates(s):
                    try:
                        return json.loads(cand)
                    except Exception:
                        continue
                try:
                    return json.loads(s)
                except Exception:
                    pass
                if strict:
                    return None
                for c in balanced_json_candidates(s):
                    if not c:
                        continue
                    try:
                        return json.loads(c)
                    except Exception:
                        pass
                return None

            def strip_ansi(text: str):
                return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text or "")

            def normalize_generation_output(text: str):
                s = strip_ansi(text or "").replace("\r", "\n")
                lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
                noisy_prefixes = (
                    "ggml_", "llama_", "common_", "build:", "main:", "sampling:", "load:",
                    "system_info", "cpu :", "gpu :", "n_threads", "model loaded", "cuda",
                )
                filtered = []
                for ln in lines:
                    low = ln.lower()
                    if "loading model" in low:
                        continue
                    if low.startswith("build"):
                        continue
                    if low.startswith("model"):
                        continue
                    if low.startswith("modalities"):
                        continue
                    if low.startswith("available commands"):
                        continue
                    if ln.startswith("/"):
                        continue
                    if low.startswith("/exit") or low.startswith("/quit") or low.startswith("/help") or low.startswith("/regen"):
                        continue
                    if ln.startswith(">"):
                        continue
                    if ln.startswith("[") and ("prompt:" in low or "generation:" in low):
                        continue
                    if low.startswith("exiting"):
                        continue
                    if low.startswith("device ") or low.startswith("compute capability") or "vram:" in low:
                        continue
                    if low.startswith("total vram") or low.startswith("main:"):
                        continue
                    if any(ch in ln for ch in ("▄", "█", "▀", "▁", "▂", "▃")):
                        continue
                    if any(low.startswith(p) for p in noisy_prefixes):
                        continue
                    filtered.append(ln)
                return "\n".join(filtered).strip()

            def fenced_json_candidates(text: str):
                cands = []
                for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE):
                    block = (m.group(1) or "").strip()
                    if block:
                        cands.append(block)
                cands.sort(key=len, reverse=True)
                return cands

            def balanced_json_candidates(text: str):
                candidates = []
                stack = []
                start = None
                for i, ch in enumerate(text):
                    if ch in "{[":
                        if not stack:
                            start = i
                        stack.append(ch)
                        continue
                    if ch in "}]":
                        if not stack:
                            continue
                        opener = stack[-1]
                        if (opener == "{" and ch == "}") or (opener == "[" and ch == "]"):
                            stack.pop()
                            if not stack and start is not None:
                                candidates.append(text[start:i + 1])
                                start = None
                        else:
                            stack.clear()
                            start = None
                candidates.sort(key=lambda c: (text.rfind(c), len(c)), reverse=True)
                return candidates

            def check_with_reason(text: str, kind: str):
                if kind == "exact_refuse":
                    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
                    if not lines:
                        return False, "parse"
                    return lines[0].lower().startswith("refuse"), "semantic"
                obj = extract_json(text, strict=(kind == "tool_args"))
                if obj is None:
                    return False, "parse"
                if kind == "json_object":
                    return (isinstance(obj, dict) and "name" in obj and "age" in obj), "semantic"
                if kind == "single_tool":
                    return (isinstance(obj, dict) and obj.get("tool") and isinstance(obj.get("args"), dict)), "semantic"
                if kind == "multi_tool":
                    return (isinstance(obj, list) and len(obj) >= 2 and all(isinstance(x, dict) and x.get("tool") for x in obj[:2])), "semantic"
                if kind == "tool_args":
                    return (
                        isinstance(obj, dict)
                        and obj.get("tool") == "email_send"
                        and isinstance(obj.get("args"), dict)
                        and all(k in obj["args"] for k in ("to", "subject", "body"))
                    ), "semantic"
                return False, "semantic"

            out_rows = []
            for c in cases:
                cmd = [
                    _LLAMA_CLI,
                    "-m", model_path,
                    "-c", str(ctx),
                    "-ngl", "999",
                    "--temp", "0.0",
                    "--top-k", "1",
                    "--seed", "42",
                    "--single-turn",
                    "--simple-io",
                    "--no-display-prompt",
                    "-p", c["prompt"],
                    "-n", "64",
                ] + kv_args(kv)
                try:
                    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
                    txt = normalize_generation_output((proc.stdout or "") + "\n" + (proc.stderr or ""))
                    ok, fail_kind = check_with_reason(txt, c["check"])
                    out_rows.append(
                        {
                            "id": c["id"],
                            "check": c["check"],
                            "pass": bool(ok),
                            "fail_kind": (None if ok else fail_kind),
                            "output_head": txt[:160],
                        }
                    )
                except Exception as exc:
                    out_rows.append({"id": c["id"], "check": c["check"], "pass": False, "error": f"{type(exc).__name__}: {exc}"})

            passed = sum(1 for r in out_rows if r.get("pass"))
            total = len(out_rows)
            failed = total - passed
            pass_rate = round((passed / total) if total else 0.0, 4)
            error_count = sum(1 for r in out_rows if r.get("error"))
            parse_fail_count = sum(1 for r in out_rows if (not r.get("pass")) and (not r.get("error")) and r.get("fail_kind") == "parse")
            semantic_fail_count = sum(1 for r in out_rows if (not r.get("pass")) and (not r.get("error")) and r.get("fail_kind") == "semantic")
            failed_checks = [str(r.get("id")) for r in out_rows if not r.get("pass")]

            if error_count >= 2:
                diagnosis = "harness_limited"
                status = "needs_harness_fix"
            elif passed <= 1:
                diagnosis = "model_limited"
                status = "not_agent_ready"
            elif passed <= 3:
                diagnosis = "mixed"
                status = "partial_agent_readiness"
            else:
                diagnosis = "config_ready_for_smoke"
                status = "smoke_pass"

            return json.dumps({
                "schema": "sigilant.agent_smoke.v1",
                "backend": "modal",
                "quant_label": data.get("quant_label"),
                "passed": passed,
                "total": total,
                "failed": failed,
                "pass_rate": pass_rate,
                "status": status,
                "diagnosis": diagnosis,
                "error_count": error_count,
                "parse_fail_count": parse_fail_count,
                "semantic_fail_count": semantic_fail_count,
                "failed_checks": failed_checks,
                "results": out_rows,
                "note": (
                    "Smoke only. Use full Sigilant Optimizer for production agent safety, "
                    "robustness, and long-context reliability certification."
                ),
            })

        payload = json.dumps({
            "model_repo": model_repo,
            "model_filename": model_filename,
            "context": int(context),
            "kv_type": kv_type,
            "quant_label": quant_label,
        })

        with modal.enable_output():
            with app.run():
                raw = smoke_job.remote(payload)
        return json.loads(raw) if raw else None


# ── serialisation helpers ────────────────────────────────────────────────────

def _config_to_dict(c: RunConfig) -> dict:
    return {
        "quant_label":    c.quant_label,
        "context":        c.context,
        "batch":          c.batch,
        "kv_type":        c.kv_type,
        "regime":         c.regime,
        "depth_label":    c.depth_label,
        "model_repo":     c.model_repo,
        "model_filename": c.model_filename,
    }


def _dict_to_result(config: RunConfig, d: dict) -> RunResult:
    if d.get("error"):
        return RunResult(config=config, error=d["error"], preflight=d.get("preflight"))
    return RunResult(
        config=config,
        tps=d.get("tps", 0.0),
        tps_p95=d.get("tps_p95"),
        ttft_ms=d.get("ttft_ms", 0.0),
        ttft_p95_ms=d.get("ttft_p95_ms"),
        itl_ms=d.get("itl_ms", 0.0),
        ppl=d.get("ppl"),
        preflight=d.get("preflight"),
    )
