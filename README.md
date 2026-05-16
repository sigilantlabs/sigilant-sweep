# sigilant-sweep

Open-source LLM inference sweep. Measure TPS, TTFT, ITL, and PPL across 16 configurations on your own hardware — local GPU, Modal, or RunPod.

```
sigilant-sweep · Mistral-7B-Instruct-v0.3 · RTX 4090 24GB · llama.cpp · 16 configs

Config                                      TPS     TTFT    ITL     PPL    Score
──────────────────────────────────────────────────────────────────────────────────
Q5_K_M · ctx:16384 · kv:f16   · b:4        53.3    612ms   19.2ms  8.44   91  ← best
Q5_K_M · ctx:8192  · kv:f16   · b:4        53.1    609ms   19.1ms  8.44   89
Q4_K_M · ctx:16384 · kv:f16   · b:4        56.2    591ms   18.1ms  8.71   87
Q4_K_M · ctx:8192  · kv:f16   · b:4        55.8    594ms   18.3ms  8.71   85
... 12 more configs

Best config:  Q5_K_M · ctx:16384 · kv:f16 · b:4

PPL is a quality proxy, not production validation.

! Agent safety NOT evaluated.
  Structural JSON, tool calling, hallucination resistance,
  and prompt injection are not covered by this sweep.

  → sigilantlabs.com/optimize
```

---

## Install

```bash
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

## Quick wow path (2 minutes)

```bash
sigilant-sweep run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --score-profile balanced \
  --agent-smoke
```

You get:
- ranked configs with deterministic winner
- baseline delta line (speed and latency uplift)
- `sigilant_results.json`, `sigilant_summary.md`, `sigilant_frontier.svg`
- smoke diagnosis (`model_limited` vs `harness_limited` vs `mixed`)

Confidence guardrails:
- Default is fixed `--trials 12` for stronger stability out of the box.
- You can override `--trials` manually for faster/cheaper or deeper runs.
- Artifacts include confidence inputs: top-2 gap and variance proxy.

---

## Hardware options

| Flag                       | Where it runs          |
|----------------------------|------------------------|
| `--backend local`          | Your machine (default) |
| `--backend modal`          | Modal cloud (your account) |
| `--backend runpod`         | RunPod cloud (your account) |

| `--hardware` value  | GPU              | VRAM  |
|---------------------|------------------|-------|
| `auto`              | auto-detect      | —     |
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
sigilant-sweep deploy --backend runpod     # builds + deploys worker image (one-time)
export SIGILANT_RUNPOD_ENDPOINT_ID=<printed-endpoint-id>
sigilant-sweep run --model mistralai/Mistral-7B-Instruct-v0.3 --backend runpod --engine llama.cpp --hardware rtx4090
```

---

## What this measures

| Metric | Description |
|--------|-------------|
| **TPS** | Output tokens per second |
| **TTFT** | Time to first token (ms) |
| **ITL** | Inter-token latency (ms) |
| **PPL** | Perplexity on a fixed corpus — lightweight quality proxy |
| **Score** | Sigilant composite (preset-based): balanced/latency/quality profiles |

## What this does NOT measure

- Tool calling correctness
- Structured JSON / schema output validity
- Hallucination resistance
- Prompt injection resistance
- Long-context retrieval (NIAH)

PPL catches gross quantization degradation. It does not validate production agent safety.

## vLLM status

- Implemented:
  - local vLLM sweep
  - Modal vLLM sweep (HF model localized at run start and reused through the sweep)
- Not implemented yet:
  - RunPod vLLM backend
  - vLLM agent smoke

PPL corpus quality note:
- Current PPL corpus is intentionally lightweight and should be treated as a coarse proxy.
- For close winners, a small/synthetic corpus can under-separate configs.
- Use higher trials for stability, and treat PPL as directional unless you swap in a larger, domain-representative corpus.

Boundary:
- OSS `sigilant-sweep`: fast config recommendation and lightweight smoke triage.
- Paid [Sigilant Optimizer](https://sigilantlabs.com/optimize): full safety/quality gates, long-context reliability, and deployment-grade certification.

### Score profiles

- `balanced`: `40% TPS + 20% TTFT + 40% PPL`
- `latency`: `50% TPS + 30% TTFT + 20% PPL`
- `quality`: `30% TPS + 20% TTFT + 50% PPL`

If PPL is unavailable, TPS/TTFT weights are renormalized automatically.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
