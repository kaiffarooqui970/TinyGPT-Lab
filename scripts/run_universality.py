"""Phase 9: feature universality across model scale + SAE quality vs model size.

Compares SAEs trained on S3, S4, S5 (and a second-seed S5 SAE as the ceiling) on the same
held-out tokens, and writes figures and JSON under report/.

Usage:
    uv run python scripts/run_universality.py --saes s3=<sae.pt> s4=<sae.pt> \
        s5=<sae.pt> s5b=<sae.pt>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tinygpt.data.prepare import resolve
from tinygpt.interp.features import context_of
from tinygpt.interp.universality import latent_matrix, max_correlation, summarise
from tinygpt.sae.collect import ActivationStore
from tinygpt.sae.train_sae import load_sae
from tinygpt.tokenizer.bpe import BPETokenizer
from tinygpt.train.metrics import pick_device


def heldout(sae_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """The exact held-out tokens the SAE was evaluated on (end of its training budget)."""
    cfg = {
        k.strip(): v.strip()
        for k, v in (
            line.split(":", 1)
            for line in (sae_path.parent / "config.yaml").read_text().splitlines()
            if ":" in line
        )
    }
    store = ActivationStore(cfg["acts_dir"])
    max_t = None if cfg.get("max_tokens") in (None, "null") else int(cfg["max_tokens"])
    acts, toks = store.load_all(max_tokens=max_t)
    h = int(cfg["heldout_tokens"])
    return acts[-h:], toks[-h:]


@torch.no_grad()
def token_identity_corr(
    z: torch.Tensor, toks: torch.Tensor, vocab: int, device: torch.device
) -> np.ndarray:
    """Per latent: max over tokens v of corr(z, 1[token == v]); 1.0 = a pure token detector."""
    zf = z.to(device).float()
    n = zf.shape[0]
    mean = zf.mean(0)
    sd = zf.std(0).clamp_min(1e-8)
    sums = torch.zeros(vocab, zf.shape[1], device=device)
    sums.index_add_(0, toks.to(device), zf)
    p = torch.bincount(toks.to(device), minlength=vocab).float() / n
    cov = sums / n - p[:, None] * mean[None, :]
    corr = cov / (sd[None, :] * torch.sqrt(p * (1 - p)).clamp_min(1e-12)[:, None])
    corr[p == 0] = -1
    best = corr.max(0).values.cpu().numpy()
    best[(zf.std(0) == 0).cpu().numpy()] = np.nan
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--saes", nargs="+", required=True, help="name=path/to/sae.pt")
    ap.add_argument("--out", default="report")
    ap.add_argument("--tokenizer", default="data/tokenizer/tinystories_8k.json")
    ap.add_argument(
        "--n-tokens",
        type=int,
        default=65_536,
        help="held-out tokens used for matching (dense latents are n x latents)",
    )
    args = ap.parse_args()
    device = pick_device()
    out = resolve(args.out)
    paths = {k: resolve(v) for k, v in (s.split("=", 1) for s in args.saes)}
    tok = BPETokenizer.load(resolve(args.tokenizer))

    Z, quality, tokens_ref, tokid = {}, {}, None, {}
    for name, p in paths.items():
        sae, ck = load_sae(p, device)
        acts, toks = heldout(p)
        acts, toks = acts[: args.n_tokens], toks[: args.n_tokens]
        if tokens_ref is None:
            tokens_ref = toks
        elif not torch.equal(tokens_ref, toks):
            raise RuntimeError(f"{name}: held-out tokens differ; comparisons need identical tokens")
        Z[name] = latent_matrix(sae, acts, device=device)
        q = json.loads((p.parent / "final.json").read_text())
        quality[name] = {
            k: q[k]
            for k in (
                "fve",
                "l0",
                "dead_frac",
                "dense_frac",
                "ce_clean",
                "ce_sae",
                "ce_delta",
                "ce_recovered",
                "n_latents",
            )
        }
        quality[name]["d_model"] = ck["acts_meta"]["d_model"]
        quality[name]["layer"] = ck["acts_meta"]["layer"]
        tokid[name] = token_identity_corr(Z[name], toks, tok.vocab_size, device)

    pairs = [("s5b", "s5"), ("s4", "s5"), ("s3", "s5"), ("s3", "s4"), ("s5", "s3"), ("s5", "s4")]
    pairs = [(a, b) for a, b in pairs if a in Z and b in Z]
    mc, summ = {}, {}
    for a, b in pairs:
        r = max_correlation(Z[a], Z[b], device=device)
        mc[(a, b)] = r
        summ[f"{b}_matched_in_{a}"] = summarise(r["max_corr"])

    # Scale-specific S5 features: reproducible across S5 seeds but unmatched at smaller scale.
    special: dict[str, object] = {}
    if all(k in mc for k in [("s5b", "s5"), ("s4", "s5"), ("s3", "s5")]):
        seed = mc[("s5b", "s5")]["max_corr"]
        m4, m3 = mc[("s4", "s5")]["max_corr"], mc[("s3", "s5")]["max_corr"]
        ok = np.isfinite(seed) & np.isfinite(m4) & np.isfinite(m3)
        reproducible = ok & (seed >= 0.7)
        only_large = reproducible & (np.maximum(m3, m4) < 0.3)
        shared = reproducible & (np.minimum(m3, m4) >= 0.7)
        ti = tokid["s5"]
        special = {
            "n_reproducible": int(reproducible.sum()),
            "n_only_s5": int(only_large.sum()),
            "n_shared_all": int(shared.sum()),
            "frac_only_s5_of_reproducible": float(only_large.sum() / max(1, reproducible.sum())),
            "token_identity_corr_only_s5_median": float(np.nanmedian(ti[only_large]))
            if only_large.any()
            else None,
            "token_identity_corr_shared_median": float(np.nanmedian(ti[shared]))
            if shared.any()
            else None,
            "freq_only_s5_median": float(
                np.median((Z["s5"][:, only_large] > 0).float().mean(0).numpy())
            )
            if only_large.any()
            else None,
            "freq_shared_median": float(np.median((Z["s5"][:, shared] > 0).float().mean(0).numpy()))
            if shared.any()
            else None,
        }
        # Examples of S5-only features, for the write-up.
        rng = np.random.default_rng(0)
        ids = np.flatnonzero(only_large)
        pick = rng.choice(ids, size=min(12, ids.size), replace=False) if ids.size else []
        lines = []
        T = 256
        for i in sorted(pick.tolist()):
            col = Z["s5"][:, i].float()
            top = torch.topk(col, 5).indices.tolist()
            lines.append(
                f"## S5 latent {i}  seed-corr={seed[i]:.2f}  best S4={m4[i]:.2f}  "
                f"best S3={m3[i]:.2f}  token-identity={ti[i]:.2f}"
            )
            for r in top:
                ids_, start, tgt = context_of(r, tokens_ref, T, before=20, after=3)
                s = "".join(
                    tok.decode([t]) if j != tgt else "[[" + tok.decode([t]) + "]]"
                    for j, t in enumerate(ids_)
                )
                lines.append(f"  {float(col[r]):.2f} | " + s.replace("\n", "⏎"))
            lines.append("")
        (out / "results" / "universality_examples.md").write_text("\n".join(lines))

    # Raw arrays for scripts/make_figures.py (plots are made from these logged results).
    arrays = {f"maxcorr_{a}_{b}": r["max_corr"] for (a, b), r in mc.items()}
    arrays.update({f"tokid_{k}": v for k, v in tokid.items()})
    arrays.update({f"freq_{k}": (z > 0).float().mean(0).numpy() for k, z in Z.items()})
    np.savez(out / "results" / "universality_arrays.npz", **arrays)

    res = {
        "quality": quality,
        "matching": summ,
        "scale_specific": special,
        "token_identity_median": {k: float(np.nanmedian(v)) for k, v in tokid.items()},
    }
    (out / "results" / "universality.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
