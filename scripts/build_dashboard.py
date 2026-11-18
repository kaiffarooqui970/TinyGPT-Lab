"""Compute feature statistics for a trained SAE and build the static HTML dashboard,
plus a plain-text label sheet for the randomly sampled features to hand-label.

Usage:
    uv run python scripts/build_dashboard.py --sae runs/sae/<sweep>/<run>/sae.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tinygpt.data.prepare import resolve
from tinygpt.interp.dashboard import build_feature_records, write_dashboard
from tinygpt.interp.features import compute_feature_stats, logit_effects
from tinygpt.model.gpt import load_gpt
from tinygpt.sae.collect import ActivationStore
from tinygpt.sae.train_sae import load_sae
from tinygpt.tokenizer.bpe import BPETokenizer
from tinygpt.train.metrics import pick_device
from tinygpt.utils.seed import seed_everything


def load_labels(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with open(path) as fh:
        return {int(r["feature"]): r for r in csv.DictReader(fh) if r.get("label")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--tokens", type=int, default=2_000_000, help="tokens used for stats")
    ap.add_argument("--n-label", type=int, default=50)
    ap.add_argument("--n-extra", type=int, default=350)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="report")
    ap.add_argument("--tokenizer", default="data/tokenizer/tinystories_8k.json")
    args = ap.parse_args()
    seed_everything(args.seed)
    device = pick_device()
    out = resolve(args.out)
    sae, ck = load_sae(args.sae, device)
    meta = ck["acts_meta"]
    store = ActivationStore(_acts_dir(args.sae))
    acts, toks = store.load_all(max_tokens=args.tokens)
    stats = compute_feature_stats(sae, acts, n_top=20, device=device)
    interp_dir = Path(args.sae).parent
    np.savez(
        interp_dir / "feature_stats.npz",
        freq=stats.freq,
        mean_active=stats.mean_active,
        top_vals=stats.top_vals,
        top_rows=stats.top_rows,
        hist=stats.hist,
    )

    alive = np.flatnonzero(stats.freq > 0)
    rng = np.random.default_rng(args.seed)
    to_label = sorted(rng.choice(alive, size=min(args.n_label, alive.size), replace=False).tolist())
    rest = np.setdiff1d(alive, to_label)
    extra = sorted(rng.choice(rest, size=min(args.n_extra, rest.size), replace=False).tolist())
    model, _ = load_gpt(resolve(meta["checkpoint"]), device)
    tok = BPETokenizer.load(resolve(args.tokenizer))
    feats = sorted(set(to_label) | set(extra))
    effects = logit_effects(model, sae.cpu(), feats)
    labels_path = out / "results" / "feature_labels.csv"
    labels = load_labels(labels_path)
    recs = build_feature_records(
        feats, stats, sae.cpu(), acts, toks, meta["block_size"], tok, effects, labels=labels
    )
    summary = {
        "SAE": Path(args.sae).parent.name,
        "latents": sae.n_latents,
        "alive (on stats tokens)": int(alive.size),
        "stats tokens": int(len(acts)),
        "layer": meta["layer"],
        "features shown": len(feats),
    }
    (out / "dashboard").mkdir(parents=True, exist_ok=True)
    write_dashboard(out / "dashboard" / "features.html", recs, "tinygpt-lab SAE features", summary)

    # Label sheet: compact text for the sampled features.
    by_id = {r["id"]: r for r in recs}
    lines = []
    for i in to_label:
        r = by_id[i]
        lines.append(f"## feature {i}  freq={r['freq']:.4%}  mean={r['mean']:.2f}")
        lines.append("boost: " + " ".join(repr(t) for t, _ in r["boost"][:8]))
        lines.append("suppress: " + " ".join(repr(t) for t, _ in r["suppress"][:5]))
        for e in r["examples"][:8]:
            ctx = (
                "".join(e["toks"][: e["target"]])
                + "[["
                + e["toks"][e["target"]]
                + "]]"
                + "".join(e["toks"][e["target"] + 1 :])
            )
            lines.append(f"  {e['max']:.2f} | " + ctx.replace("\n", "⏎"))
        lines.append("")
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / "label_sheet.md").write_text("\n".join(lines))
    (out / "results" / "label_sample.json").write_text(
        json.dumps({"features": to_label, "seed": args.seed, "sae": args.sae}, indent=1)
    )
    print(json.dumps({**summary, "label_sample": to_label}))


def _acts_dir(sae_path: str) -> str:
    cfg = (Path(sae_path).parent / "config.yaml").read_text()
    return next(
        line.split(":", 1)[1].strip() for line in cfg.splitlines() if line.startswith("acts_dir:")
    )


if __name__ == "__main__":
    main()
