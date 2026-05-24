![Sigilant Sweep Banner](banner.png)

# sigilant-sweep

Benchmark orchestration for inference stacks (llama.cpp, vLLM): TPS, TTFT, ITL, PPL proxy, and artifacted comparisons.

[![PyPI](https://img.shields.io/pypi/v/sigilant-sweep?style=for-the-badge)](https://pypi.org/project/sigilant-sweep/)
[![License](https://img.shields.io/badge/license-Apache%202.0-1f6feb?style=for-the-badge)](https://github.com/sigilantlabs/sigilant-sweep/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/sigilantlabs/sigilant-sweep?style=for-the-badge)](https://github.com/sigilantlabs/sigilant-sweep/stargazers)

[Scope](#scope) • [Install](#install) • [First-time success](#first-time-success-guide) • [Metrics](#what-this-measures) • [Reproducibility](#verification-and-reproducibility)

---
## Scope

`sigilant-sweep` is orchestration and reporting around existing inference engines.

It handles:
- config generation
- benchmark execution via adapters (`llama.cpp`, `vllm`)
- metric parsing (TPS, TTFT, ITL, PPL proxy)
- scoring and artifact export

It is not a new inference runtime.

## Non-goals

- custom kernels or scheduler innovation
- replacing engine internals (`llama.cpp`, `vllm`)
- claiming production safety certification from throughput benchmarks

---

## Install

```bash
# Refresh installer tooling first (recommended)
python3 -m pip install -U pip

# Base (lightweight CLI + reporting)
pip install sigilant-sweep

# Hugging Face integration only
pip install 'sigilant-sweep[hf]'

# With llama.cpp
pip install 'sigilant-sweep[llama]'

# With llama.cpp + CUDA acceleration
CMAKE_ARGS="-DGGML_CUDA=on" pip install 'sigilant-sweep[llama]'

# With vLLM (Linux + CUDA only)
pip install 'sigilant-sweep[vllm]'

# With Modal cloud backend
pip install 'sigilant-sweep[modal]'

# With RunPod cloud backend
pip install 'sigilant-sweep[runpod]'

# Everything
pip install 'sigilant-sweep[all]'
```

If your environment uses a custom package index or stale mirror, force PyPI:

```bash
pip install --index-url https://pypi.org/simple sigilant-sweep
```

---

## First-time success guide

### Golden path: Modal (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip setuptools wheel
pip install "sigilant-sweep[modal]"
modal token new
sigilant-sweep info
```

Run a cheap sanity test:

```bash
sigilant-sweep run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 1 \
  --trials 1
```

### Golden path: Local llama.cpp

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip setuptools wheel
pip install sigilant-sweep
```

Requirements:
- `llama-cli` must be installed and discoverable on `PATH`, or set `SIGILANT_LLAMA_CLI=/abs/path/to/llama-cli`.
- Local backend is compute-dependent; on CPU-only machines it will be slow.

### Compatibility matrix (current recommendation)

| Scenario | Recommended install | Notes |
|---|---|---|
| Any OS, Modal-only | `pip install "sigilant-sweep[modal]"` | Best first-run success path |
| Any OS, HF-only | `pip install "sigilant-sweep[hf]"` | For model listing/download integration |
| Local llama.cpp | `pip install sigilant-sweep` | Requires external `llama-cli` binary |
| Local vLLM | `pip install "sigilant-sweep[vllm]"` | Linux + CUDA only |

### Known install issue (Intel macOS + Modal extras)

If you see `Failed building wheel for cbor2`:

```bash
pip uninstall -y modal cbor2
pip install --only-binary=:all: "cbor2==5.6.5"
pip install "sigilant-sweep[modal]"
```

Then verify:

```bash
python3 -c "import modal, cbor2; print('modal', modal.__version__, 'cbor2_ok', hasattr(cbor2, 'dumps'))"
```

---

## Quick start

```bash
# 1. Check hardware and credentials
sigilant-sweep setup

# 2. Show what's detected on this machine
sigilant-sweep info

# 3. Run a sweep (local GPU, llama.cpp)
sigilant-sweep run --model mistralai/Mistral-7B-Instruct-v0.3

# 4. Save results to JSON
sigilant-sweep run --model mistralai/Mistral-7B-Instruct-v0.3 --json
```

## Modal run example

```bash
sigilant-sweep run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --score-profile balanced \
  --agent-smoke
```

Output:
- ranked configs with score and status
- baseline delta line
- `sigilant_results.json`, `sigilant_summary.md`, `sigilant_frontier.svg`
- optional smoke diagnosis (`model_limited` vs `harness_limited` vs `mixed`)

Stability notes:
- Default is fixed `--trials 12` for stronger stability out of the box.
- You can override `--trials` manually for faster/cheaper or deeper runs.
- Artifacts include confidence inputs: top-2 gap and variance proxy.

## Common run patterns

Single config only:

```bash
sigilant-sweep run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 16 \
  --trials 1 \
  --only-config "Q4_K_M,8192,k16v16,default"
```

Depth profile:

```bash
sigilant-sweep run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 16 \
  --trials 5 \
  --benchmark-mode depth_profile \
  --depth-prompt-8k prompts/hard_quality_8k_prompt.txt \
  --depth-prompt-14k prompts/hard_quality_14k_prompt.txt \
  --depth-prompt-28k prompts/hard_quality_28k_prompt.txt
```

Run with smoke check:

```bash
sigilant-sweep run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 16 \
  --trials 5 \
  --agent-smoke
```

## Execution model

- CLI resolves model files, builds the config grid, dispatches to backend, and scores results.
- llama.cpp path runs timed generation and perplexity per config/trial, then aggregates (`p50`, `p95`, `mean PPL`).
- Multi-trial runs are rotated trial-first to avoid running all trials of one config back-to-back.
- Artifacts are written under `artifacts/runs/<run_id>/`.

## Troubleshooting

- `Model resolution failed: huggingface-hub is required`
: install `pip install "sigilant-sweep[hf]"` or `pip install "sigilant-sweep[modal]"`.

- `Error: modal is not installed`
: install `pip install "sigilant-sweep[modal]"`.

- `Version ... of modal is deprecated`
: upgrade modal in venv: `pip install -U modal`.

- `Failed building wheel for cbor2` (Intel macOS path)
: run
`pip uninstall -y modal cbor2 && pip install --only-binary=:all: "cbor2==5.6.5" && pip install "sigilant-sweep[modal]"`.

- vLLM local failures on macOS/Windows
: expected; use Modal backend for vLLM.

## Release checklist (clean run)

Run this sequence exactly from repo root.

### 1) Preflight

```bash
source .venv/bin/activate
bash scripts/release_preflight.sh <new_version>
```

This checks:
- active directory is repo root
- required files exist
- `sigilant_runner/__init__.py` derives version from package metadata
- version equals target argument

### 2) Commit release changes

```bash
git add README.md pyproject.toml
git commit -m "release: bump to <new_version>"
git push origin main
```

### 3) Build, upload, index-check, fresh-venv verify (single command)

```bash
bash scripts/release_verify.sh <new_version>
```

`release_verify.sh` runs:
- preflight
- clean build (`dist/`, `build/`)
- twine check + upload for exact target artifacts
- PyPI simple-index polling until target is visible
- fresh-venv install check (`pip show`, `sigilant-sweep --version`)

### 4) Runtime sanity

llama.cpp Modal:

```bash
sigilant-sweep run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 1 \
  --trials 1
```

vLLM Modal:

```bash
export SIGILANT_VLLM_FAMILY_REPOS='{"FP16_BASELINE":"microsoft/Phi-3.5-mini-instruct"}'
sigilant-sweep run \
  --model microsoft/Phi-3.5-mini-instruct \
  --backend modal \
  --engine vllm \
  --hardware a10g \
  --configs 1 \
  --trials 1
```

---

## Hardware options

| Flag                       | Where it runs          |
|----------------------------|------------------------|
| `--backend local`          | Your machine (default) |
| `--backend modal`          | Modal cloud (your account) |
| `--backend runpod`         | RunPod cloud (your account) |

| `--hardware` value  | GPU              | VRAM  |
|---------------------|------------------|-------|
| `auto`              | auto-detect      | n/a   |
| `a10g`              | NVIDIA A10G      | 24 GB |
| `a100`              | NVIDIA A100      | 40 GB |
| `h100`              | NVIDIA H100      | 80 GB |
| `l4`                | NVIDIA L4        | 24 GB |
| `t4`                | NVIDIA T4        | 16 GB |
| `rtx4090`           | RTX 4090         | 24 GB |
| `rtx3090`           | RTX 3090         | 24 GB |
| `rtxa6000`          | RTX A6000        | 48 GB |

---

## Engine options

| Flag                 | Supported Backends           | Notes |
|----------------------|------------------------------|-------|
| `--engine llama.cpp` | `local`, `modal`, `runpod`   | GGUF-based flow |
| `--engine vllm`      | `local`, `modal`             | Linux + CUDA required |

---

## Full CLI reference

```
sigilant-sweep run [OPTIONS]

  --model      -m    HuggingFace repo ID or local .gguf path   [required]
  --backend    -b    local | modal | runpod                     [default: local]
  --engine     -e    llama.cpp | vllm                           [default: llama.cpp]
  --hardware         GPU target (see table above)               [default: auto]
  --params-b         Model size in billions (for VRAM estimate) [default: 7.0]
  --configs          Max number of configs to sweep             [default: 16]
  --confidence-target  low | medium | high                      [default: medium] (reporting only)
  --score-profile      balanced | latency | quality             [default: balanced]
  --trials             Trials per config                        [default: 12]
  --json             Also write results to sigilant_results.json

sigilant-sweep setup    Check credentials for all backends (interactive)
sigilant-sweep info     Show detected hardware and installed engines
sigilant-sweep --version
```

---

## Cloud backend setup

### Modal

```bash
pip install 'sigilant-sweep[modal]'
modal token new          # saves credentials to ~/.modal.toml
sigilant-sweep run --model mistralai/Mistral-7B-Instruct-v0.3 --backend modal --hardware a10g
```

### RunPod

```bash
pip install 'sigilant-sweep[runpod]'
export RUNPOD_API_KEY=<your-key>
export SIGILANT_RUNPOD_ENDPOINT_ID=<your-predeployed-endpoint-id>
sigilant-sweep run --model mistralai/Mistral-7B-Instruct-v0.3 --backend runpod --engine llama.cpp --hardware rtx4090
```

---

## What this measures

| Metric | Description |
|--------|-------------|
| **TPS** | Output tokens per second |
| **TTFT** | Time to first token (ms) |
| **ITL** | Inter-token latency (ms) |
| **PPL** | Perplexity on a fixed corpus, used as a lightweight quality proxy |
| **Score** | Sigilant composite (preset-based): balanced/latency/quality profiles |

## What this does NOT measure

- Tool calling correctness
- Structured JSON / schema output validity
- Hallucination resistance
- Prompt injection resistance
- Long-context retrieval (NIAH)

PPL catches gross quantization degradation. It does not validate production agent safety.

Prompt corpus note:
- Prompt and corpus files in `prompts/` are benchmark assets maintained for this harness.
- They are intended for relative configuration comparison, not as a standardized external evaluation set.

## Verification and reproducibility

- Keep raw artifacts with reported tables (`sigilant_results.json`, `sigilant_terminal.txt`).
- Re-run top candidates with `--only-config` before final selection.
- Separate infra/control-plane failures from model/runtime failures.
- Treat PPL as a ranking proxy within comparable runs.

## vLLM status

- Implemented:
  - local vLLM sweep
  - Modal vLLM sweep (HF model localized at run start and reused through the sweep)
- Not implemented yet:
  - RunPod vLLM backend
  - vLLM structured-output smoke

PPL corpus quality note:
- Current PPL corpus is intentionally lightweight and should be treated as a coarse proxy.
- For close winners, a small/synthetic corpus can under-separate configs.
- Use higher trials for stability, and treat PPL as directional unless you swap in a larger, domain-representative corpus.

Boundary:
- OSS `sigilant-sweep`: config ranking, runtime metrics, and lightweight smoke triage.
- For broader capability/safety validation on production workloads, use [Sigilant Optimizer](https://sigilantlabs.com/optimize).

### Score profiles

- `balanced`: `40% TPS + 20% TTFT + 40% PPL`
- `latency`: `50% TPS + 30% TTFT + 20% PPL`
- `quality`: `30% TPS + 20% TTFT + 50% PPL`

If PPL is unavailable, TPS/TTFT weights are renormalized automatically.

---

## License

Apache 2.0. See [LICENSE](LICENSE).
