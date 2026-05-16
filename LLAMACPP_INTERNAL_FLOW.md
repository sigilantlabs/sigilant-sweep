# llama.cpp Internal Flow (sigilant-sweep)

This document describes the exact runtime flow for `sigilant-sweep run --engine llama.cpp` in this repo.

## 1. CLI entry

File: `sigilant_runner/cli.py`

1. Parse CLI flags (`--model`, `--backend`, `--hardware`, `--configs`, `--trials`, `--score-profile`, etc.).
2. Resolve model references (quant variants + filenames).
3. Generate config grid (usually 16 configs).
4. Dispatch to backend:
   - local: `sigilant_runner/backends/local.py`
   - modal: `sigilant_runner/backends/modal_backend.py`
5. Receive `RunResult[]`.
6. Compute score/normalization in `sigilant_runner/core/scoring.py`.
7. Print terminal table and write artifacts.
8. Optional: run agent smoke (`--agent-smoke`) on winner.

## 2. Modal backend boot

File: `sigilant_runner/backends/modal_backend.py`

1. Build Modal image:
   - CUDA base image
   - compile `llama.cpp` binaries:
     - `llama-cli`
     - `llama-perplexity`
     - `llama-bench`
     - `llama-server`
2. Start app function `benchmark_sweep`.
3. Load benchmark payload:
   - configs
   - trials
   - benchmark prompt
   - PPL corpus text

## 3. PPL corpus source resolution

Current precedence in `ModalBackend.run()`:

1. `SIGILANT_PPL_CORPUS` file (if set and exists)
2. shared file:
   - `prompts/hard_quality_8k_prompt.txt`
3. bundled fallback:
   - `prompts/ppl_corpus_250.txt` in this repo

The selected text is embedded into payload and re-used in the worker.

## 4. Model localization

Inside `benchmark_sweep`:

1. For each unique `(model_repo, model_filename)`, call `hf_hub_download`.
2. Cache downloaded path in `file_cache`.
3. Every config maps to one of these local model paths.

## 5. Trial execution semantics (rotated, trial-first)

This repo now runs trials in **trial-first rotated order**.

Given:
- `n_cfg = len(configs)` (usually 16)
- `n_trials = max(1, trials)`
- `step = max(1, n_cfg // n_trials)`

For each trial `t`:
1. `start = (t * step) % n_cfg`
2. Build trial order:
   - `configs[start:] + configs[:start]`
3. Run **each config once** in that order.

This avoids running all trials of one config back-to-back.

## 6. Per-config single trial work

For each config execution:

1. Build kv args:
   - `k8v8` => `--cache-type-k q8_0 --cache-type-v q8_0`
   - `k16v16` => default (no kv args)
2. Run `llama-cli` with:
   - fixed decode params (temp=0, top-k=1, seed=42)
   - fixed token generation length (`-n 128`)
3. Parse metrics from stdout/stderr:
   - TPS
   - TTFT
   - ITL
4. If TTFT or ITL missing, apply fallbacks:
   - TTFT from wall-time minus decode-time estimate
   - ITL from `1000 / TPS`
5. If parse invalid, mark trial error for that config.

## 7. Per-trial PPL computation

PPL is computed via `llama-perplexity` and logged per trial:

1. Write payload corpus to temp file.
2. Choose effective PPL eval context:
   - requested: `SIGILANT_PPL_EVAL_CTX` (default 2048)
   - downshift if corpus estimated tokens are too small.
3. Run:
   - `llama-perplexity -m <model> -ngl 999 -c <ppl_ctx> -f <temp_file> [kv args]`
4. Parse strict pattern:
   - `Final estimate: PPL = ...`
5. Save full PPL command + stdout/stderr in trial logs.

Important:
- If perplexity reports insufficient tokens, PPL stays `None` and reason is recorded.

## 8. Aggregation per config

After all trials:

For each config:
1. `tps` = median(`tps_vals`)
2. `ttft_ms` = median(`ttft_vals`)
3. `itl_ms` = median(`itl_vals`)
4. `tps_p95` = percentile 95 over trial TPS values
5. `ttft_p95_ms` = percentile 95 over trial TTFT values
6. `ppl` = mean over successful `ppl_vals` collected across trials

If config has no successful trials:
- output row is `FAILED` with error + preflight trial logs.

## 9. Score computation

File: `sigilant_runner/core/scoring.py`

Current behavior:
1. Build normalized throughput score from **TPS p95** (fallback to p50 if p95 missing).
2. Build normalized latency score from **TTFT p95** (fallback to p50 if p95 missing).
3. Build quality score from PPL (lower is better): `min_ppl / ppl`.
4. Weighted score by profile:
   - balanced: 40 / 20 / 40
   - latency: 50 / 30 / 20
   - quality: 30 / 20 / 50
5. If PPL missing globally, renormalize to TPS/TTFT only.

## 10. Artifacts produced

Per run folder (`artifacts/runs/<run_id>/`):

1. `sigilant_results.json`
2. `sigilant_summary.md`
3. `sigilant_frontier.svg`
4. `sigilant_terminal.txt`

`sigilant_results.json` contains:
- config-level metrics and score
- status/error
- `preflight.trial_logs[]` with raw commands and stdout/stderr tails/full text

## 11. Agent smoke flow

1. CLI checks winner config.
2. Local backend: uses local `llama-cli`.
3. Modal backend: `ModalBackend.run_agent_smoke(...)` launches dedicated smoke function.
4. Runs 5 lightweight checks:
   - structural JSON
   - single tool call
   - multi-tool call
   - refusal behavior
   - tool args shape
5. Produces `passed/total`, diagnosis, and status.

## 12. Common failure modes and meaning

1. `PPL = None` with `insufficient_corpus_tokens_for_ctx`
   - corpus too short for chosen perplexity context.
2. `timing_parse_failed`
   - llama output format changed / parser mismatch.
3. `llama-cli rc != 0`
   - runtime failure (OOM, invalid args, binary issue).
4. `all trials failed`
   - no successful trial for that config.

## 13. Quick verification checklist

For any run:

1. Confirm trials rotated:
   - logs should show `Trial x/y start=c...`.
2. Confirm p95 present:
   - `tps_p95`, `ttft_p95_ms` non-null in `sigilant_results.json`.
3. Confirm PPL source:
   - logs print selected corpus and char count.
4. Confirm PPL validity:
   - `llama_ppl_stderr` contains `Final estimate: PPL = ...`.
5. Confirm failures are explicit:
   - each failed row has `error` and `preflight.trial_logs`.
