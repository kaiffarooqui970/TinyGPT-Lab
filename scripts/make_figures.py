"""Regenerate every report figure from logged runs (no training or evaluation).

Usage:
    uv run python scripts/make_figures.py                       # full study -> report/
    uv run python scripts/make_figures.py --config configs/tiny/figures.yaml
"""

from __future__ import annotations

import argparse
import json

from tinygpt.figures import FiguresConfig, make_all
from tinygpt.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/figures.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    summary = make_all(load_config(args.config, FiguresConfig, args.set))
    print(json.dumps(summary, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
