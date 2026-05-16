# llama.cpp on OSS/GitHub

This guide is for users who clone the repo and run `llama.cpp` sweeps on Modal.

## 1) Install

```bash
git clone https://github.com/<org>/sigilant-sweep.git
cd sigilant-sweep
python3.11 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip setuptools wheel
pip install ".[modal]"
modal token new
```

If install fails with `Failed building wheel for cbor2`, run:

```bash
pip uninstall -y modal cbor2
pip install --only-binary=:all: "cbor2==5.6.5"
pip install ".[modal]"
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
