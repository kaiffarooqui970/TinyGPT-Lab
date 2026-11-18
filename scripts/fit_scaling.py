"""Fit the scaling law from a completed sweep and write figures + a results table.

Usage:
    uv run python scripts/fit_scaling.py --sweep scaling --out report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinygpt.scaling.analyze import scaling_analysis


def table(res: dict) -> str:
    f = res["fit_N"]
    c = f["ci95"]
    rows = [
        "| parameter | estimate | 95% CI (eval-batch bootstrap) |",
        "|---|---|---|",
        f"| E (irreducible loss) | {f['E']:.3f} | [{c['E'][0]:.3f}, {c['E'][1]:.3f}] |",
        f"| A | {f['A']:.4g} | [{c['A'][0]:.4g}, {c['A'][1]:.4g}] |",
        f"| α | {f['alpha']:.3f} | [{c['alpha'][0]:.3f}, {c['alpha'][1]:.3f}] |",
    ]
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default="scaling")
    ap.add_argument("--out", default="report")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out)
    res = scaling_analysis(args.sweep, out / "figures", n_boot=args.n_boot)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / "scaling_fit.json").write_text(json.dumps(res, indent=2))
    (out / "results" / "scaling_fit.md").write_text(table(res) + "\n")
    print(table(res))
    print(
        json.dumps({k: res[k] for k in ("N", "val_loss", "rmse", "extrapolation_test")}, indent=1)
    )


if __name__ == "__main__":
    main()
