# Sigilant Runner OSS Status (Handoff)
Date: 2026-05-04  
Repo: `sigilant-runner`  
Scope: OSS lightweight optimizer only (no edits to `sigilant-autotune` runtime flow)

## 1) Current Product Shape
`sigilant-runner` is positioned as a lightweight, open-source inference config optimizer:
- Sweeps a fixed grid (target 16 configs)
- Scores configs on speed + quality proxy:
  - TPS (p50/p95)
  - TTFT (p50/p95)
  - PPL
- Produces winner + artifacts (`json`, `md`, `svg`, terminal snapshot)
- Optional agent smoke checks (lightweight only)

It is intentionally not a full replacement for Sigilant paid optimizer phases.

## 2) What Has Been Achieved

### 2.1 Core scoring/outputs
- Stable CLI output table with:
  - `TPS p50`, `TPS p95`
  - `TTFT p50`, `TTFT p95`
  - `ITL`
  - `PPL`
  - normalized % columns + final score
- Score profile presets are available (`balanced`, `latency`, `quality`).
- Baseline compare support exists (auto and explicit baseline).
- Artifact export is implemented:
  - `sigilant_results.json`
  - `sigilant_summary.md`
  - `sigilant_frontier.svg`
  - `sigilant_terminal.txt`

### 2.2 Llama.cpp path
- Llama.cpp path is operational and remains the stable path.
- PPL parsing was hardened vs earlier failures.
- Multi-trial runs and rotation logic are in place.
- Quant selection policy includes fallback behavior where needed.

### 2.3 Charting/UX artifacts
- Frontier SVG generator significantly improved from earlier raw plot.
- Legend/annotation style is now cleaner and adaptable to run context.
- Terminal output persistence by run is implemented in artifacts folder.

### 2.4 vLLM wiring (major progress)
- vLLM execution path exists in `ModalBackend` (`benchmark_sweep_vllm`).
- Model localization happens at run start (repo snapshot local caching).
- Family-level repo override support exists.
- Added preflight metadata and better startup error tail capture.
- Quantization inference patch added:
  - if HF model config already declares quantization, do **not** force CLI quantization.

### 2.5 vLLM family redesign (latest)
The grid family names were changed from old labels to explicit OSS target labels:
1. `FP16_BASELINE`
2. `INT8_W8A8`
3. `AWQ4_MARLIN`
4. `GPTQ4_MARLIN`

Related code updates were applied in:
- `sigilant_runner/models.py`
- `sigilant_runner/backends/modal_backend.py`
- `sigilant_runner/engines/vllm_engine.py`
- `sigilant_runner/cli.py` (`_quant_bits`)
- `sigilant_runner/output/export.py` (labels/colors)

## 3) Where We Are Stuck

### 3.1 FP8 repo failure root cause is now known
Observed failure for `RedHatAI/Phi-3.5-mini-instruct-FP8-KV` on L4/A10 path:
- `torch._inductor` / Triton error during vLLM engine init
- unsupported dtype variant (`fp8e4nv` not supported in this runtime/architecture path)

This is **not** the old mismatch bug (`--quantization fp8` vs `compressed-tensors`) anymore.
That mismatch was fixed by inference behavior.

### 3.2 vLLM config failure visibility was incomplete
Previously errors were too shallow; now improved, but still needs one final cleanup pass:
- Ensure every failed config writes full startup preflight + concise root-cause code.
- Ensure `preflight` appears consistently for both success and fail rows.

### 3.3 Family repos are user-dependent
vLLM families now require valid HF repos per family bucket:
- FP16 base repo
- INT8/W8A8 repo
- AWQ4 repo
- GPTQ4 repo

If missing or incompatible, startup fails by design.

## 4) Pending Work (Priority Order)

### P0 (must finish first)
1. Validate latest vLLM family rename end-to-end with one run.
2. Confirm env var overrides resolve correctly for all 4 families:
   - `SIGILANT_VLLM_FP16_BASELINE_REPO`
   - `SIGILANT_VLLM_INT8_W8A8_REPO`
   - `SIGILANT_VLLM_AWQ4_MARLIN_REPO`
   - `SIGILANT_VLLM_GPTQ4_MARLIN_REPO`
3. Confirm results show up as 16 configs (4 families × 4 ctx/kv combos) when repos are valid.

### P1
4. Improve vLLM startup diagnostics formatting in result JSON:
   - Add `failure_code` field (e.g., `UNSUPPORTED_FP8_DTYPE`, `OOM`, `REPO_QUANT_MISMATCH`)
   - Keep raw error tail but include short normalized cause.
5. Add a pre-run “family repo sanity” phase:
   - Download + read `config.json`
   - quick quant method compatibility gate
   - fail-fast before full sweep when repo is clearly wrong.

### P2
6. vLLM runtime efficiency:
   - reduce restart overhead where safe
   - improve seq-candidate heuristics by hardware/model size
7. Documentation hardening in README for vLLM family repo requirements.

## 5) Files Most Relevant for Next Debug Session
- `sigilant_runner/backends/modal_backend.py`
  - `benchmark_sweep_vllm`
  - `_family_profile`
  - `_resolved_quantization_arg`
  - `_family_supported_for_repo`
  - startup + error capture path
- `sigilant_runner/models.py`
  - vLLM family list
- `sigilant_runner/core/grid.py`
  - 4 combo generation per family
- `sigilant_runner/cli.py`
  - baseline logic, quant-bits mapping, run orchestration
- `sigilant_runner/output/export.py`
  - artifact serialization + frontier rendering

## 6) Known Good / Known Bad Summary

### Known good
- Llama.cpp flow for OSS runs
- Artifact generation pipeline
- Quantization mismatch fix (do not force fp8 when repo declares quantization config)
- Improved visibility into vLLM startup failures

### Known bad / unresolved
- FP8 compressed-tensors repo (`RedHatAI/Phi-3.5-mini-instruct-FP8-KV`) fails on current L4/A10 runtime path due to dtype support in engine compile path.
- End-to-end proof run for newly renamed vLLM family buckets still pending.

## 7) Recommended Next Run Plan (Low GPU Waste)
1. Start with single-family validation:
   - `SIGILANT_VLLM_FAMILIES=FP16_BASELINE`
   - verify 4/4 configs run
2. Add `INT8_W8A8` with known compatible int8/w8a8 repo.
3. Add AWQ4 and GPTQ4 only after individual smoke validation.
4. Run full 16 config only after all four family smokes pass.

## 8) Hard Constraints to Preserve
- Do not change or break llama.cpp existing behavior.
- Keep Modal + artifact contract stable.
- Keep output artifacts backward-readable (`sigilant_results.json` remains primary source).

## 9) Context Note for New Codex Tab
When continuing in a new tab:
- Start by reading this file and latest failed run artifact.
- Validate vLLM family repos before any broad run.
- Do not spend GPU on full sweep before one-family smoke passes.
