# vLLM on OSS/GitHub

This guide is for users who clone the repo and run vLLM sweeps on Modal.

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

## 2) Set vLLM family repos

```bash
export SIGILANT_VLLM_INT8_W8A8_REPO=anhbn/Phi-3.5-mini-instruct-quantized.w8a8
export SIGILANT_VLLM_AWQ4_MARLIN_REPO=thesven/Phi-3.5-mini-instruct-awq
export SIGILANT_VLLM_GPTQ4_MARLIN_REPO=thesven/Phi-3.5-mini-instruct-GPTQ-4bit
```

## 3) Run vLLM on Modal (ranking only)

```bash
sigilant-sweep run \
  --model microsoft/Phi-3.5-mini-instruct \
  --backend modal \
  --engine vllm \
  --hardware l4 \
  --configs 16 \
  --trials 1 \
  --benchmark-mode ranking
```

## 4) Single-config debug

```bash
sigilant-sweep run \
  --model microsoft/Phi-3.5-mini-instruct \
  --backend modal \
  --engine vllm \
  --hardware l4 \
  --configs 16 \
  --trials 1 \
  --only-config "INT8_W8A8,32768,k8v8,long"
```

## 5) 32k stress (single config + long prompt)

```bash
SIGILANT_BENCH_PROMPT_FILE=prompts/hard_quality_28k_prompt.txt \
sigilant-sweep run \
  --model microsoft/Phi-3.5-mini-instruct \
  --backend modal \
  --engine vllm \
  --hardware l4 \
  --configs 16 \
  --trials 1 \
  --only-config "INT8_W8A8,32768,k8v8,long"
```
