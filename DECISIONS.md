# Design decisions

One line of reasoning per decision. Newest at the bottom of each phase.

## Phase 0 — Setup

- **Python 3.12 (pinned in `.python-version`)**: the newest version with mature PyTorch wheels on all platforms; the system has 3.15, which torch support lags behind.
- **Repo at `~/tinygpt-lab`, not on the Desktop**: iCloud Desktop sync evicts venv files, which makes imports hang.
- **src layout + hatchling editable install**: stops tests from importing the working directory by accident, so they exercise the installed package.
- **Extra modules `utils/config.py` and `utils/run.py`** (not in the PLAN layout): config validation and run-dir creation are separate concerns from seeding and logging; keeping each module small follows the "small focused modules" rule.
- **Configs are dataclasses validated by a small hand-written `from_dict`**: unknown keys, missing fields and wrong types fail at load time, with no pydantic dependency (not on the allowed list).
- **`bool` is rejected where an `int` is expected; `int` and numeric strings are accepted for `float`**: YAML parses `1e-3` as a string and `1` as an int, and silently treating `True` as 1 hides typos.
- **Run dir = `runs/<YYYYmmdd-HHMMSS-micro>_<name>/` with `config.yaml`, `meta.json` (git hash, dirty flag, argv, versions), `metrics.jsonl`**: everything needed to trace a number back to code + config; `mkdir` without `exist_ok`, so runs are never overwritten.
- **`git_dirty` recorded alongside the hash**: a hash alone is misleading if the run used uncommitted code.
- **JSONL logger flushes every line; NaN/inf logged as strings**: crashed runs keep their logs; JSON has no NaN, and dropping a diverged value would hide the divergence.
- **`seed_everything(deterministic=False)` by default**: deterministic kernels slow CUDA training; they are switched on only for the bit-exact resume tests.
- **wandb is an optional extra (`uv sync --extra wandb`)**: the JSONL logs are the source of truth for every reported number.
- **pre-commit runs ruff + ruff-format + pytest**: lint and the test suite gate every commit.
- **pre-commit hooks call `uv run ruff`/`uv run pytest` (local hooks)**: the hook versions then match `uv.lock` exactly; a separately pinned ruff disagreed on deprecated rules.
