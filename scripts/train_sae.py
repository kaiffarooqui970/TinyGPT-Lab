"""Train one SAE, or a sweep of SAEs, on collected activations.

Usage:
    uv run python scripts/train_sae.py --config configs/sae/base.yaml --set k=32
    uv run python scripts/train_sae.py --sweep configs/sae/sweep_s5.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import yaml

from tinygpt.data.prepare import resolve
from tinygpt.sae.train_sae import SAETrainConfig, train_sae
from tinygpt.utils.config import apply_overrides, from_dict, load_config
from tinygpt.utils.logging import JSONLLogger


@dataclass
class SAESweepConfig:
    name: str
    base_config: str
    runs: dict[str, list[str]]  # run name -> overrides
    overrides: list[str] | None = None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--sweep")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    if args.config:
        _, m = train_sae(load_config(args.config, SAETrainConfig, args.set))
        print(json.dumps(m, indent=1))
        return
    sw = load_config(args.sweep, SAESweepConfig)
    base = yaml.safe_load(resolve(sw.base_config).read_text())
    root = resolve("runs") / "sae" / sw.name
    rows = []
    for name, sets in sw.runs.items():
        if args.only and name not in args.only:
            continue
        done = [d for d in sorted(root.glob(f"*_{name}")) if (d / "final.json").exists()]
        if done:
            m = json.loads((done[-1] / "final.json").read_text())
        else:
            data = apply_overrides(
                json.loads(json.dumps(base)),
                [
                    *(sw.overrides or []),
                    *sets,
                    f"name={name}",
                    f"runs_root=runs/sae/{sw.name}",
                    *args.set,
                ],
            )
            _, m = train_sae(from_dict(SAETrainConfig, data))
        rows.append({"run": name, **m})
        print(
            json.dumps(
                {
                    "run": name,
                    **{
                        k: m[k]
                        for k in ("l0", "fve", "ce_delta", "ce_recovered", "dead_frac")
                        if k in m
                    },
                }
            ),
            flush=True,
        )
    with JSONLLogger(root / "summary.jsonl") as log:
        for r in rows:
            log.log(r)


if __name__ == "__main__":
    main()
