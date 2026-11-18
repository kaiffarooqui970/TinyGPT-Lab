"""Tokenize a dataset into uint16 memmap shards (train/val) and report token counts.

Usage:
    uv run python scripts/prepare_data.py --config configs/data/tinystories.yaml
"""

from __future__ import annotations

import argparse

from tinygpt.data.prepare import PrepareConfig, tokenize_split, write_meta
from tinygpt.utils.config import load_config
from tinygpt.utils.logging import JSONLLogger
from tinygpt.utils.run import create_run_dir
from tinygpt.utils.seed import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config, PrepareConfig)
    seed_everything(cfg.seed)
    run_dir = create_run_dir(cfg, name="prepare_data")
    counts = {split: tokenize_split(cfg, split) for split in ("val", "train")}
    write_meta(cfg, counts)
    with JSONLLogger(run_dir / "metrics.jsonl") as log:
        log.log({"train_tokens": counts["train"], "val_tokens": counts["val"]})
    print(counts)
    print(run_dir)


if __name__ == "__main__":
    main()
