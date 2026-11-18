# tinygpt-lab

A GPT-style language model built from scratch in plain PyTorch, plus a scaling-law study
and sparse-autoencoder interpretability on its activations. See `PLAN.md` for the full plan
and `DECISIONS.md` for design choices.

**Status:** Phase 0 (setup) complete. There are no results yet; every number will come from logged runs.

## Setup

```bash
uv sync                       # creates .venv with Python 3.12
uv run pre-commit install     # ruff + pytest on every commit
uv run pytest                 # run the test suite
```

## Smoke test

```bash
uv run python scripts/hello_run.py --config configs/hello.yaml
# -> runs/<timestamp>_hello/{config.yaml, meta.json, metrics.jsonl}
```
