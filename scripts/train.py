"""Train a GPT from a YAML config, optionally resuming from a checkpoint.

Usage:
    uv run python scripts/train.py --config configs/train/s5_10m.yaml
    uv run python scripts/train.py --config configs/train/tiny_cpu.yaml --set lr=3e-3
    uv run python scripts/train.py --resume runs/<run>/ckpt_last.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinygpt.train.trainer import TrainConfig, Trainer
from tinygpt.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[], help="overrides like lr=1e-3 model.n_layer=4")
    ap.add_argument("--resume", help="checkpoint to resume; its run dir's config is used")
    args = ap.parse_args()
    if args.resume:
        ckpt = Path(args.resume)
        cfg = load_config(ckpt.parent / "config.yaml", TrainConfig, args.set)
        trainer = Trainer(cfg, run_dir=ckpt.parent, resume_from=ckpt)
    else:
        cfg = load_config(args.config, TrainConfig, args.set)
        trainer = Trainer(cfg)
    print(
        f"run dir: {trainer.run_dir}  N_nonembed={trainer.n_params:,}  "
        f"steps={trainer.total_steps}  device={trainer.device}",
        flush=True,
    )
    result = trainer.train()
    trainer.close()
    print(json.dumps({k: v for k, v in result.items() if k != "val_loss_per_batch"}))


if __name__ == "__main__":
    main()
