"""Smoke test for the experiment plumbing: config -> seed -> run dir -> JSONL metrics.

Usage:
    uv run python scripts/hello_run.py --config configs/hello.yaml
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from tinygpt.utils.config import load_config
from tinygpt.utils.logging import JSONLLogger
from tinygpt.utils.run import REPO_ROOT, create_run_dir
from tinygpt.utils.seed import seed_everything


@dataclass
class HelloConfig:
    """Config for the hello run."""

    name: str = "hello"
    seed: int = 0
    steps: int = 5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs-root", default=str(REPO_ROOT / "runs"))
    args = ap.parse_args()

    cfg = load_config(args.config, HelloConfig)
    seed_everything(cfg.seed)
    run_dir = create_run_dir(cfg, root=args.runs_root, name=cfg.name)
    with JSONLLogger(run_dir / "metrics.jsonl") as log:
        for step in range(cfg.steps):
            log.log({"step": step, "value": random.random()})
    print(run_dir)


if __name__ == "__main__":
    main()
