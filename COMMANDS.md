# Commands (llama.cpp)

## Setup once

```bash
cd /path/to/sigilant-sweep
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

## Run locally (quick sanity)

```bash
sigilant-sweep run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend local \
  --engine llama.cpp \
  --configs 16 \
  --trials 3 \
  --score-profile balanced
```

## Run on Modal (L4)

```bash
sigilant-sweep run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 16 \
  --trials 5 \
  --score-profile balanced
```

## Run on Modal (A10G, quality-focused)

```bash
sigilant-sweep run \
  --model bartowski/Phi-3.5-mini-instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware a10g \
  --configs 16 \
  --trials 5 \
  --score-profile quality \
  --agent-smoke
```

## Pin a baseline config (optional)

```bash
sigilant-sweep run \
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --backend modal \
  --engine llama.cpp \
  --hardware l4 \
  --configs 16 \
  --trials 5 \
  --baseline-config "Q8_0|8192|k16v16|default"
```

## Artifacts

```text
artifacts/runs/<timestamp>/
```

Main files:

- `sigilant_results.json`
- `sigilant_summary.md`
- `sigilant_frontier.svg`
- `sigilant_terminal.txt`

## Small operating rules

- Use `--trials 3` for speed, `--trials 8+` for stability.
- Keep `--engine llama.cpp` for llama.cpp runs.
- Keep `--backend` as `local` or `modal`.
