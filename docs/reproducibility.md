# Reproducibility

## Recommended protocol

1. Run full ranking once.
2. Re-run top candidates with `--only-config`.
3. Keep all run artifacts and command lines.
4. Record environment details (backend, hardware, engine, package version).

## Minimal commands

### llama.cpp (Modal, 1-config sanity)

```bash
sigilant-sweep run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 1 \
  --trials 1
```

### vLLM (Modal, fixed single-config check)

```bash
SIGILANT_VLLM_INT8_W8A8_REPO=anhbn/Phi-3.5-mini-instruct-quantized.w8a8 \
sigilant-sweep run \
  --model microsoft/Phi-3.5-mini-instruct \
  --backend modal \
  --engine vllm \
  --hardware l4 \
  --configs 16 \
  --trials 1 \
  --only-config "INT8_W8A8,32768,k8v8,long"
```

## Artifact checks

- Confirm `status` per config row (`pass` vs `failed_*`).
- Inspect `sigilant_terminal.txt` for infra/runtime errors.
- Confirm reported run settings in `sigilant_results.json` context block.

