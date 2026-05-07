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

## 1) Normal run (balanced, local)

```bash
sigilant-runner run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend local \
  --engine llama.cpp \
  --configs 16 \
  --trials 10 \
  --score-profile balanced
```

---

## 2) Normal run (quality profile, local)

```bash
sigilant-runner run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend local \
  --engine llama.cpp \
  --configs 16 \
  --trials 10 \
  --score-profile quality
```

---

## 3) Depth profile run (8k / 14k / 28k, local)

```bash
sigilant-runner run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend local \
  --engine llama.cpp \
  --configs 16 \
  --trials 10 \
  --score-profile balanced \
  --benchmark-mode depth_profile \
  --depth-prompt-8k prompts/hard_quality_8k_prompt.txt \
  --depth-prompt-14k prompts/hard_quality_14k_prompt.txt \
  --depth-prompt-28k prompts/hard_quality_28k_prompt.txt
```

---

## 4) Normal run + Agent smoke (local)

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

---

## 5) Depth profile + Agent smoke (local)

```bash
sigilant-runner run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend local \
  --engine llama.cpp \
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

## 6) Local quick sanity

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

## 7) Higher confidence run (local)

```bash
sigilant-runner run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend local \
  --engine llama.cpp \
  --configs 16 \
  --trials 20 \
  --score-profile balanced \
  --benchmark-mode depth_profile
```

---

## 8) Modal equivalents (optional)

Add these flags to any command above when you want Modal instead of local:

```bash
--backend modal --hardware l4
```

---

## 9) Output location

Artifacts are written under:

`artifacts/runs/<run_id>/`

Typical files:
- `sigilant_results.json`
- `sigilant_summary.md`
- `sigilant_frontier.svg`
- `sigilant_terminal.txt`
