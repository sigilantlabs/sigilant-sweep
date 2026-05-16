# Limitations

## Benchmark scope limits

This tool is optimized for comparative configuration ranking, not full capability validation.

Not covered:
- tool-calling correctness
- JSON/schema reliability
- hallucination resistance
- jailbreak/prompt-injection resistance
- end-to-end product safety gates

## Metric limits

- PPL is a quality proxy and should be interpreted relatively within comparable runs.
- Short prompts can under-stress long-context settings; use depth prompts for context stress.
- Single-trial runs are useful for smoke checks, not robust ranking.

## Infra limits

- Cloud control-plane issues (timeouts/heartbeat/network) can produce infra failures independent of model quality.
- Engine startup and backend availability may vary by region/provider.

## Interpretation guidance

Use this ordering for decisions:
1. run validity and failure classification
2. TTFT and decode throughput tradeoff
3. PPL proxy and workload-specific validation

