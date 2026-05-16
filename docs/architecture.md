# Architecture

## Purpose

`sigilant-sweep` is a benchmark orchestration layer over existing engines.

## Pipeline

1. CLI parses run options (`model`, `backend`, `engine`, `configs`, `trials`, profile flags).
2. Model resolver maps user model input to engine-specific families/quant targets.
3. Config builder creates benchmark grid (context/kv/regime variants).
4. Backend adapter executes runs (`local`, `modal`, `runpod`) using selected engine path.
5. Engine parser extracts runtime metrics (TPS, TTFT, ITL, optional PPL proxy).
6. Scoring normalizes metrics and applies profile weights.
7. Exporter writes artifacts (`json`, `summary`, `frontier`, terminal snapshot).

## Engine boundary

- Inference is executed by external engines (`llama.cpp`, `vllm`).
- This repo does not modify model internals, kernels, or scheduler logic.

## Artifact contract

Per run:
- `artifacts/runs/<run_id>/sigilant_results.json`
- `artifacts/runs/<run_id>/sigilant_summary.md`
- `artifacts/runs/<run_id>/sigilant_frontier.svg`
- `artifacts/runs/<run_id>/sigilant_terminal.txt`

