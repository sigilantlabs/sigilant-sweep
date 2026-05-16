# PyPI Naming Plan (Public OSS)

## Target public identity

- Repo: `sigilant-sweep`
- PyPI package: `sigilant-sweep`
- CLI command: `sigilant-sweep`

## Current implementation choice

- Internal Python module remains `sigilant_runner` for now (safe migration path).
- `pyproject.toml` package name is set to `sigilant-sweep`.
- CLI script now exposes:
  - `sigilant-sweep` (primary)
  - `sigilant-runner` (compatibility alias)

## User command (public)

```bash
pip install sigilant-sweep
sigilant-sweep run --model ... --backend modal --engine llama.cpp
```

## Follow-up before publish

1. Update README examples to use `sigilant-sweep` as primary command.
2. Keep `sigilant-runner` alias for one release cycle, then optionally remove.
3. Tag release and publish to PyPI under `sigilant-sweep`.
