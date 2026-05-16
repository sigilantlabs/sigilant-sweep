# llama.cpp on OSS/GitHub

This guide is for users who clone the repo and run `llama.cpp` sweeps.

## 1) Install

```bash
git clone https://github.com/<org>/sigilant-sweep.git
cd sigilant-sweep
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -e .
pip install "sigilant-sweep[modal]"
modal token new
```

## 2) Run llama.cpp on Modal (ranking only)

```bash
sigilant-sweep run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 16 \
  --trials 1 \
  --benchmark-mode ranking
```

## 3) Optional depth profile

```bash
sigilant-sweep run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 16 \
  --trials 1 \
  --benchmark-mode depth_profile \
  --depth-prompt-8k prompts/hard_quality_8k_prompt.txt \
  --depth-prompt-14k prompts/hard_quality_14k_prompt.txt \
  --depth-prompt-28k prompts/hard_quality_28k_prompt.txt
```

## 4) Single-config debug

```bash
sigilant-sweep run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 16 \
  --trials 1 \
  --only-config "Q4_K_M,16384,k16v16,long"
```
