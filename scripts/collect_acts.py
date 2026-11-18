"""Collect residual-stream activations from a checkpoint into fp16 shards.

Usage:
    uv run python scripts/collect_acts.py --config configs/collect/s5_l3.yaml
"""

from __future__ import annotations

import argparse
import json

from tinygpt.sae.collect import CollectConfig, collect
from tinygpt.utils.config import load_config
from tinygpt.utils.logging import JSONLLogger
from tinygpt.utils.run import create_run_dir
from tinygpt.utils.seed import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, CollectConfig, args.set)
    seed_everything(cfg.seed)
    run_dir = create_run_dir(cfg, name="collect")
    meta = collect(cfg)
    with JSONLLogger(run_dir / "metrics.jsonl") as log:
        log.log({k: v for k, v in meta.items() if k != "norm_by_position"})
    print(
        json.dumps(
            {k: v for k, v in meta.items() if k not in ("norm_by_position", "shard_sizes")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
