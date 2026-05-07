# Sigilant Runner

GGUF config optimizer for `llama.cpp` with reproducible benchmarking on `local` and `modal`.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Recommended](https://img.shields.io/badge/recommended-python%203.11%2B-2ea44f)
![Engine](https://img.shields.io/badge/engine-llama.cpp-2ea44f)
![Backends](https://img.shields.io/badge/backends-local%20%7C%20modal-7a3cff)
![Modes](https://img.shields.io/badge/modes-ranking%20%7C%20depth_profile-1f6feb)
![Status](https://img.shields.io/badge/status-vLLM%20coming%20soon-f59e0b)

This repo currently focuses on `llama.cpp`. vLLM and additional backends are planned in later releases.

## Quick Links

- [What This Does](#what-this-does)
- [Path A: Local Quick Start](#path-a-local-quick-start)
- [Path B: Modal Quick Start](#path-b-modal-quick-start)
- [Depth Profile](#depth-profile)
- [Agent Smoke](#agent-smoke-5-check-quick-gate)
- [Troubleshooting](#troubleshooting)
- [Appendix A: Install llama.cpp / llama-cli](#appendix-a-install-llamacpp--llama-cli)

## What This Does

For each config, runner measures:
- `TPS` (tokens/sec)
- `TTFT` (time to first token)
- `ITL` (inter-token latency)
- `PPL` (quality proxy)

It runs a config grid (default 16), aggregates trials, ranks results, and writes artifacts.

## Before You Start

- Python `3.10+` is supported.
- Python `3.11+` is recommended (smoother dependency installs, especially for Modal on Intel macOS).
- If you run local backend, you need `llama-cli` available.

## Path A: Local Quick Start

Use this if you want to run on your own machine (no Modal needed).

```bash
git clone https://github.com/sigilantlabs/sigilant-sweep.git
cd sigilant-sweep
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -e .
pip install -U huggingface_hub
export HF_TOKEN=hf_xxx
export SIGILANT_PPL_CORPUS=prompts/ppl_corpus_hard_mixed_6k.txt
llama-cli --version
```

If `llama-cli --version` fails:

```bash
export SIGILANT_LLAMA_CLI=/absolute/path/to/llama-cli
```

Run:

```bash
sigilant-runner run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend local \
  --engine llama.cpp \
  --configs 16 \
  --trials 3 \
  --score-profile balanced
```

## Path B: Modal Quick Start

Use this if you want cloud GPU runs.

```bash
git clone https://github.com/sigilantlabs/sigilant-sweep.git
cd sigilant-sweep
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -e .
pip install -U modal huggingface_hub
modal setup
export HF_TOKEN=hf_xxx
export SIGILANT_PPL_CORPUS=prompts/ppl_corpus_hard_mixed_6k.txt
```

Run:

```bash
sigilant-runner run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 16 \
  --trials 10 \
  --score-profile balanced \
  --agent-smoke
```

Note:
- Modal often has a free credit tier for many users, but this policy can change.
- Modal hardware is NVIDIA datacenter/enterprise GPU inventory.

## 3 Commands To First Result (Assuming Setup Already Done)

Local:

```bash
cd sigilant-sweep
source .venv/bin/activate
sigilant-runner run --model Qwen/Qwen2.5-1.5B-Instruct-GGUF --backend local --engine llama.cpp --configs 16 --trials 3 --score-profile balanced
```

Modal:

```bash
cd sigilant-sweep
source .venv/bin/activate
sigilant-runner run --model Qwen/Qwen2.5-1.5B-Instruct-GGUF --backend modal --engine llama.cpp --hardware l4 --configs 16 --trials 3 --score-profile balanced
```

## Depth Profile

Runs three prompt-depth passes and reports bucket winners.

```bash
sigilant-runner run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 16 \
  --trials 10 \
  --score-profile balanced \
  --benchmark-mode depth_profile \
  --depth-prompt-8k prompts/hard_quality_8k_prompt.txt \
  --depth-prompt-14k prompts/hard_quality_14k_prompt.txt \
  --depth-prompt-28k prompts/hard_quality_28k_prompt.txt
```

Output includes:
- `best_at_8k`
- `best_at_14k`
- `best_at_28k`
- per-bucket result tables

## Agent Smoke (5-check quick gate)

Enable with:

```bash
--agent-smoke
```

Checks:
- structural JSON
- single-tool JSON
- multi-tool JSON
- basic refusal behavior
- tool-arg JSON shape

## Scoring and Trial Semantics

- Default profile: `balanced`
- Balanced weights: `35% TPS + 25% TTFT + 40% PPL`
- TPS/TTFT normalization uses `p95` (fallback to p50 if needed)
- PPL is aggregated as mean across successful trials

Trials are **trial-first with rotated starts**:
- each trial runs all configs once
- start offset rotates per trial
- final metrics aggregate per config across trials

## Model Input

`--model` expects a Hugging Face GGUF repo.

Examples:
- `Qwen/Qwen2.5-1.5B-Instruct-GGUF`
- `Qwen/Qwen2.5-7B-Instruct-GGUF`
- `bartowski/Phi-3.5-mini-instruct-GGUF`

Split GGUF repos are supported (runner fetches sibling shards).

## Artifacts

Each run writes to:

`artifacts/runs/<run_id>/`

Files:
- `sigilant_results.json`
- `sigilant_summary.md`
- `sigilant_frontier.svg`
- `sigilant_terminal.txt`

## Troubleshooting

### 1) `modal is not installed`

You’re running `--backend modal` without modal package in this venv.

```bash
pip install -U modal
```

### 2) `huggingface-hub is required to list models`

```bash
pip install -U huggingface_hub
```

### 3) Modal install fails on Intel macOS with `cbor2` / Rust error

Typical error:
- `error: can't find Rust compiler`

Preferred fix:
- use Python `3.11+` venv for this repo and reinstall.

```bash
deactivate 2>/dev/null || true
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -e .
pip install -U modal huggingface_hub
```

### 4) All rows `FAILED`

Check:
- model repo is valid GGUF repo
- HF token available if rate-limited
- backend/hardware pairing is valid
- local path has working `llama-cli`

### 5) PPL is blank (`—`)

Most common causes:
- invalid `SIGILANT_PPL_CORPUS` path
- corpus too short for configured eval context

Try:

```bash
export SIGILANT_PPL_CORPUS=prompts/ppl_corpus_hard_mixed_6k.txt
export SIGILANT_PPL_EVAL_CTX=1536
```

### 6) Winner confidence is low

Increase trials:

```bash
--trials 15
```
or
```bash
--trials 20
```

### 7) Clean local reset

```bash
deactivate 2>/dev/null || true
rm -rf .venv
```

Then reinstall:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -e .
pip install -U huggingface_hub
```

Deleting this repo folder deletes only this repo’s `.venv`. Other virtual environments are unaffected.

## Additional Docs

- Command recipes: [COMMANDS_LLAMA_CPP.md](./COMMANDS_LLAMA_CPP.md)
- Internal execution flow: [LLAMACPP_INTERNAL_FLOW.md](./LLAMACPP_INTERNAL_FLOW.md)

## Appendix A: Install llama.cpp / llama-cli

You need `llama-cli` for local backend runs.

### Option 1: Build from source

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build -j
./build/bin/llama-cli --version
```

Then either:
- add `build/bin` to `PATH`, or
- set:

```bash
export SIGILANT_LLAMA_CLI=/absolute/path/to/llama.cpp/build/bin/llama-cli
```

### Option 2: Existing binary

If this works, no additional install is required:

```bash
llama-cli --version
```
