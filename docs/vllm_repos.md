# vLLM Repo Inventory

This file tracks the Hugging Face repos used for Sigilant Runner vLLM family buckets.

## Family Buckets

- `FP16_BASELINE`
  - Default: `microsoft/Phi-3.5-mini-instruct`
  - Env var: `SIGILANT_VLLM_FP16_BASELINE_REPO`

- `INT8_W8A8`
  - Current: `anhbn/Phi-3.5-mini-instruct-quantized.w8a8`
  - Env var: `SIGILANT_VLLM_INT8_W8A8_REPO`

- `AWQ4_MARLIN`
  - Current: `thesven/Phi-3.5-mini-instruct-awq`
  - Env var: `SIGILANT_VLLM_AWQ4_MARLIN_REPO`

- `GPTQ4_MARLIN`
  - Current: `thesven/Phi-3.5-mini-instruct-GPTQ-4bit`
  - Env var: `SIGILANT_VLLM_GPTQ4_MARLIN_REPO`

## Single-Family Smoke Commands

### AWQ only

```bash
export SIGILANT_VLLM_AWQ4_MARLIN_REPO=thesven/Phi-3.5-mini-instruct-awq
export SIGILANT_VLLM_FAMILIES=AWQ4_MARLIN
```

### GPTQ only

```bash
export SIGILANT_VLLM_GPTQ4_MARLIN_REPO=thesven/Phi-3.5-mini-instruct-GPTQ-4bit
export SIGILANT_VLLM_FAMILIES=GPTQ4_MARLIN
```

Then run:

```bash
sigilant-runner run \
  --model microsoft/Phi-3.5-mini-instruct \
  --backend modal \
  --engine vllm \
  --hardware a10g \
  --configs 4 \
  --trials 1 \
  --score-profile balanced \
  --benchmark-mode ranking
```
