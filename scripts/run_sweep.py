"""Run (or resume) a sweep defined in configs/sweeps/*.yaml.

Usage:
    uv run python scripts/run_sweep.py --config configs/sweeps/lr_sweep_s1.yaml
    uv run python scripts/run_sweep.py --config configs/sweeps/scaling.yaml --dry-run
"""

from __future__ import annotations

import argparse

from tinygpt.scaling.run_sweep import SweepConfig, plan_runs, run_sweep
from tinygpt.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", nargs="*", help="run-name prefixes to run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sweep = load_config(args.config, SweepConfig)
    if args.dry_run:
        for name, sets in plan_runs(sweep):
            print(name, " ".join(sets))
        return
    run_sweep(sweep, only=args.only)


if __name__ == "__main__":
    main()
