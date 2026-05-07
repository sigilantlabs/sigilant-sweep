# Sigilant Runner

Sigilant Runner is a GGUF config optimizer.

This repo currently focuses on:
- `llama.cpp` engine
- `local` and `modal` backends

Planned:
- add `vLLM` and additional engines/backends in later releases

It runs a grid sweep, measures latency/throughput/quality, ranks configs, and writes run artifacts.

## What You Get

- 16-config default sweep across quant/ctx/kv regimes
- multi-trial benchmarking with rotated trial starts
- `TPS p50/p95`, `TTFT p50/p95`, `ITL`, and `PPL`
- weighted score and winner selection
- optional depth profile (8k/14k/28k prompts)
- optional 5-case agent smoke gate

## Quick Start (Copy-Paste)

```bash
cd /path/to/sigilant-runner
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
modal setup
export HF_TOKEN=hf_xxx
export SIGILANT_PPL_CORPUS=prompts/ppl_corpus_hard_mixed_6k.txt

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

## Prerequisites

### 1) System

- Python `3.10+`
- `git`
- internet access to Hugging Face and Modal APIs

### 2) Repo setup

```bash
cd /path/to/sigilant-runner
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

What `pip install -e .` does:
- installs this package in editable mode
- installs Python dependencies from this repo

What it does **not** do:
- does not download GGUF models
- does not install `llama.cpp` binaries
- does not authenticate Modal

### 3) Environment variables

Recommended:
```bash
export HF_TOKEN=hf_xxx
export SIGILANT_PPL_CORPUS=prompts/ppl_corpus_hard_mixed_6k.txt
```

Optional:
```bash
export SIGILANT_PPL_EVAL_CTX=1536
```

### 4) Backend-specific requirements

#### Local backend

You must have `llama-cli` available.

Check:
```bash
llama-cli --version
```

If not in PATH:
```bash
export SIGILANT_LLAMA_CLI=/absolute/path/to/llama-cli
```

Installation options for `llama.cpp` / `llama-cli` are in [Appendix A](#appendix-a-installing-llamacpp--llama-cli).

#### Modal backend

You must authenticate Modal once on your machine:
```bash
modal setup
```

Run all modal commands from the same shell where `.venv` is active.

Note:
- Modal typically provides a monthly free credit tier for new/personal usage patterns (commonly cited as around `$30`, but policy can change).
- Modal GPU catalog is NVIDIA datacenter/enterprise-class hardware, not consumer GPUs.

## Step-by-Step: Local Run

### A) Sanity run

```bash
sigilant-runner run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend local \
  --engine llama.cpp \
  --configs 16 \
  --trials 3 \
  --score-profile balanced
```

### B) Production-strength run

```bash
sigilant-runner run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend local \
  --engine llama.cpp \
  --configs 16 \
  --trials 10 \
  --score-profile balanced \
  --agent-smoke
```

## Step-by-Step: Modal Run

### A) Sanity run

```bash
sigilant-runner run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 16 \
  --trials 3 \
  --score-profile balanced
```

### B) Production-strength run

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

### C) If memory is tight

Use stronger GPU:
```bash
--hardware a10g
```

## Depth Profile Mode

Use fixed prompt buckets to profile winners by context depth.

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

Depth output includes:
- `best_at_8k`
- `best_at_14k`
- `best_at_28k`
- full per-bucket tables

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

## Metrics and Scoring

Per config:
- TPS p50, TPS p95
- TTFT p50, TTFT p95
- ITL
- PPL (mean across successful trials)

Current `balanced` score:
- `35% TPS_norm + 25% TTFT_norm + 40% PPL_norm`

Normalization:
- `TPS_norm = TPS p95 / max TPS p95`
- `TTFT_norm = min TTFT p95 / TTFT p95`
- `PPL_norm = min PPL / PPL`

If PPL is unavailable, score is renormalized over TPS/TTFT.

## Trial Semantics (Important)

Trials are **trial-first with rotated starts**.

That means:
- each trial runs the whole config set once
- start index rotates each trial to reduce order bias
- final metrics aggregate across trials per config

This is intentionally not "run all trials for config-1, then config-2".

## Model Input Rules

`--model` expects Hugging Face GGUF repo ID.

Examples:
- `Qwen/Qwen2.5-1.5B-Instruct-GGUF`
- `Qwen/Qwen2.5-7B-Instruct-GGUF`
- `bartowski/Phi-3.5-mini-instruct-GGUF`

Split GGUF shards are handled automatically (`00001-of-0000N` siblings).

## Artifacts

Each run writes:
`artifacts/runs/<run_id>/`

Files:
- `sigilant_results.json` (raw metrics, errors, preflight)
- `sigilant_summary.md` (summary report)
- `sigilant_frontier.svg` (chart)
- `sigilant_terminal.txt` (terminal snapshot)

## Troubleshooting

### 1) All rows `FAILED`

Check:
- model repo is GGUF and accessible
- HF token is set
- backend/hardware is valid
- `llama-cli` exists (local backend)

### 2) Missing split shard error

Symptom:
- `failed to open ...-00002-of-...gguf`

Action:
- rerun once; runner now auto-fetches sibling shard set

### 3) PPL shows `—`

Most common causes:
- invalid/missing corpus file path
- corpus too short for selected eval context

Action:
```bash
export SIGILANT_PPL_CORPUS=prompts/ppl_corpus_hard_mixed_6k.txt
export SIGILANT_PPL_EVAL_CTX=1536
```

### 4) Modal warns about unauthenticated HF requests

Set:
```bash
export HF_TOKEN=hf_xxx
```

### 5) Winner confidence is low

Increase trials:
```bash
--trials 15
```
or
```bash
--trials 20
```

## Additional Docs

- Command recipes: [COMMANDS_LLAMA_CPP.md](./COMMANDS_LLAMA_CPP.md)
- Internal execution flow: [LLAMACPP_INTERNAL_FLOW.md](./LLAMACPP_INTERNAL_FLOW.md)

## Appendix A: Installing llama.cpp / llama-cli

You need a working `llama-cli` binary for local backend runs.

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
- export explicit binary path:

```bash
export SIGILANT_LLAMA_CLI=/absolute/path/to/llama.cpp/build/bin/llama-cli
```

### Option 2: Use an existing binary

If you already have `llama-cli` installed:

```bash
llama-cli --version
```

If that command works, no extra install step is required.
