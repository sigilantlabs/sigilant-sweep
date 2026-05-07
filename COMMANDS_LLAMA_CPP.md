# llama.cpp Command Recipes

All commands below are for `sigilant-runner-llamacpp`.

## 0) One-time setup (from repo root)

```bash
pip install -e .
```

Optional (recommended for download limits):

```bash
export HF_TOKEN=hf_xxx
```

Use hard mixed PPL corpus:

```bash
export SIGILANT_PPL_CORPUS=prompts/ppl_corpus_250.txt
```

---

## 1) Normal run (balanced)

```bash
sigilant-runner run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 16 \
  --trials 10 \
  --score-profile balanced
```

---

## 2) Normal run (quality profile)

```bash
sigilant-runner run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 16 \
  --trials 10 \
  --score-profile quality
```

---

## 3) Depth profile run (8k / 14k / 28k)

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

---

## 4) Normal run + Agent smoke

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

---

## 5) Depth profile + Agent smoke

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
  --depth-prompt-28k prompts/hard_quality_28k_prompt.txt \
  --agent-smoke
```

---

## 6) Local run (quick sanity)

```bash
sigilant-runner run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend local \
  --engine llama.cpp \
  --configs 16 \
  --trials 3 \
  --score-profile balanced
```

---

## 7) Higher confidence run

```bash
sigilant-runner run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 16 \
  --trials 20 \
  --score-profile balanced \
  --benchmark-mode depth_profile
```

---

## 8) Output location

Artifacts are written under:

`artifacts/runs/<run_id>/`

Typical files:
- `sigilant_results.json`
- `sigilant_summary.md`
- `sigilant_frontier.svg`
- `sigilant_terminal.txt`
