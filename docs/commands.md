# Sigilant Runner Commands

## vLLM (FP16 + INT8) Depth Profile on A10G

```bash
export SIGILANT_VLLM_INT8_W8A8_REPO=anhbn/Phi-3.5-mini-instruct-quantized.w8a8
export SIGILANT_VLLM_FAMILIES=FP16_BASELINE,INT8_W8A8

sigilant-runner run \
  --model microsoft/Phi-3.5-mini-instruct \
  --backend modal \
  --engine vllm \
  --hardware a10g \
  --configs 8 \
  --trials 1 \
  --benchmark-mode depth_profile \
  --depth-prompt-8k prompts/hard_quality_8k_prompt.txt \
  --depth-prompt-14k prompts/hard_quality_14k_prompt.txt \
  --depth-prompt-28k prompts/hard_quality_28k_prompt.txt
```

