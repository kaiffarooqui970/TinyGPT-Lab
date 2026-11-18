"""Analysis + figures for Phases 4-5: loss curves, LR sweep, scaling law L(N) and L(C)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from tinygpt.data.prepare import resolve
from tinygpt.scaling.fit import bootstrap_fit, fit_power_law, leave_one_out
from tinygpt.utils.logging import read_jsonl
from tinygpt.utils.plotting import CATEGORICAL, TEXT_2, apply_style, ordinal_blues, save


def sweep_runs(sweep: str) -> list[dict[str, Any]]:
    """Completed runs of a sweep (the latest completed dir per run name)."""
    root = resolve("runs") / "sweeps" / sweep
    latest: dict[str, Path] = {}
    for d in sorted(root.glob("*_*")):
        if (d / "final.json").exists():
            latest[d.name.split("_", 1)[1]] = d
    out = []
    for name, d in latest.items():
        res = json.loads((d / "final.json").read_text())
        cfg_text = (d / "config.yaml").read_text()
        lr = float(
            next(line.split(":", 1)[1] for line in cfg_text.splitlines() if line.startswith("lr:"))
        )
        out.append({"name": name, "dir": d, "lr": lr, **res})
    return out


# ---------------------------------------------------------------------------- LR sweep
def lr_sweep_figure(sweep: str, out_png: str | Path) -> dict[str, Any]:
    apply_style()
    runs = sorted((r for r in sweep_runs(sweep) if not r.get("diverged")), key=lambda r: r["lr"])
    lrs = np.array([r["lr"] for r in runs])
    vl = np.array([r["val_loss"] for r in runs])
    best = runs[int(vl.argmin())]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(lrs, vl, "-o", color=CATEGORICAL[0])
    ax.plot([best["lr"]], [best["val_loss"]], "o", ms=11, mfc="none", mec=CATEGORICAL[1], mew=2)
    ax.annotate(
        f"best {best['lr']:.1e}\nval {best['val_loss']:.3f}",
        (best["lr"], best["val_loss"]),
        textcoords="offset points",
        xytext=(10, 14),
        color=TEXT_2,
    )
    ax.set_xscale("log")
    ax.set_xlabel("peak learning rate")
    ax.set_ylabel("final val loss (nats/token)")
    ax.set_title(f"LR sweep at the smallest model (N = {runs[0]['n_params_nonembed']:,})")
    save(fig, out_png)
    diverged = [r["name"] for r in sweep_runs(sweep) if r.get("diverged")]
    return {
        "lrs": lrs.tolist(),
        "val_loss": vl.tolist(),
        "best_lr": best["lr"],
        "best_val_loss": best["val_loss"],
        "diverged": diverged,
    }


# ---------------------------------------------------------------------------- curves
def loss_curves_figure(run_dir: str | Path, out_png: str | Path, title: str) -> dict[str, Any]:
    apply_style()
    rows = read_jsonl(Path(run_dir) / "metrics.jsonl")
    tr = [(r["tokens"], r["train_loss"]) for r in rows if "train_loss" in r and "tokens" in r]
    tpstep = tr[0][0] / next(r["step"] for r in rows if "train_loss" in r)
    va = [(r["step"] * tpstep, r["val_loss"]) for r in rows if "val_loss" in r]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(*zip(*tr, strict=True), color=CATEGORICAL[0], lw=1.2, alpha=0.8, label="train")
    ax.plot(*zip(*va, strict=True), "-o", color=CATEGORICAL[1], ms=4, label="val")
    ax.set_xscale("log")
    ax.set_xlabel("tokens seen")
    ax.set_ylabel("loss (nats/token)")
    ax.set_title(title)
    ax.legend()
    save(fig, out_png)
    return {"final_val": va[-1][1], "final_train": tr[-1][1], "tokens": tr[-1][0]}


def multi_curves_figure(runs: list[dict[str, Any]], out_png: str | Path) -> None:
    apply_style()
    runs = sorted(runs, key=lambda r: r["n_params_nonembed"])
    colors = ordinal_blues(len(runs))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for r, c in zip(runs, colors, strict=True):
        rows = read_jsonl(r["dir"] / "metrics.jsonl")
        tr = [(row["tokens"], row["train_loss"]) for row in rows if "train_loss" in row]
        ax.plot(
            *zip(*tr, strict=True),
            color=c,
            lw=1.4,
            label=f"N = {r['n_params_nonembed'] / 1e6:.2f}M",
        )
    ax.set_xscale("log")
    ax.set_xlabel("tokens seen")
    ax.set_ylabel("train loss (nats/token)")
    ax.set_title("Training curves across model sizes (20 tokens/param)")
    ax.legend()
    save(fig, out_png)


# ---------------------------------------------------------------------------- scaling
def scaling_analysis(sweep: str, out_dir: str | Path, n_boot: int = 1000) -> dict[str, Any]:
    """Fit L(N) and L(C) with bootstrap CIs, leave-one-out; write figures + JSON."""
    apply_style()
    out_dir = Path(out_dir)
    runs = sorted(
        (r for r in sweep_runs(sweep) if not r.get("diverged")),
        key=lambda r: r["n_params_nonembed"],
    )
    # One run per size: the main grid uses the transferred LR (multiplier 1.0) only.
    by_size: dict[int, dict[str, Any]] = {}
    for r in runs:
        if r.get("is_lr_check"):
            continue
        by_size.setdefault(r["n_params_nonembed"], r)
    runs = [by_size[n] for n in sorted(by_size)]
    N = np.array([r["n_params_nonembed"] for r in runs], dtype=float)
    C = np.array([r["compute_6ND"] for r in runs], dtype=float)
    per_batch = [r["val_loss_per_batch"] for r in runs]
    L = np.array([np.mean(b) for b in per_batch])

    bN = bootstrap_fit(N, per_batch, n_boot=n_boot, seed=0)
    bC = bootstrap_fit(C, per_batch, n_boot=n_boot, seed=1)
    pred = bN.point.predict(N)
    resid = L - pred
    loo = leave_one_out(N, L)
    # Fits on the 4 smallest sizes, extrapolated to the largest: the honest predictive test.
    f4 = fit_power_law(N[:-1], L[:-1])
    extrap = {
        "fit_on": N[:-1].tolist(),
        "predict_N": float(N[-1]),
        "predicted": float(f4.predict(N[-1])),
        "actual": float(L[-1]),
        "alpha": f4.alpha,
        "E": f4.E,
    }
    sem = [float(np.std(b, ddof=1) / np.sqrt(len(b))) for b in per_batch]

    colors = ordinal_blues(len(N))
    grid = np.geomspace(N.min() / 2, N.max() * 4, 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    boot_curves = np.array([s[0] + s[1] * grid ** (-s[2]) for s in bN.samples])
    lo, hi = np.percentile(boot_curves, [2.5, 97.5], axis=0)
    ax.fill_between(
        grid, lo, hi, color=CATEGORICAL[1], alpha=0.18, lw=0, label="95% bootstrap band"
    )
    ax.plot(
        grid,
        bN.point.predict(grid),
        color=CATEGORICAL[1],
        lw=2,
        label=f"fit: {bN.point.E:.2f} + {bN.point.A:.3g}·N^-{bN.point.alpha:.3f}",
    )
    ax.axhline(bN.point.E, color=TEXT_2, lw=1, ls=":", label=f"E = {bN.point.E:.2f}")
    for n, l_, c in zip(N, L, colors, strict=True):
        ax.plot([n], [l_], "o", color=c, mec="white", mew=1.5, ms=9, zorder=3)
    ax.plot([], [], "o", color=colors[-1], label="trained models")
    ax.set_xscale("log")
    ax.set_xlabel("non-embedding parameters N")
    ax.set_ylabel("final val loss (nats/token)")
    ax.set_title("Scaling law L(N) = E + A·N^-α on TinyStories (20 tokens/param)")
    ax.legend(fontsize=8.5)
    save(fig, out_dir / "scaling_law.png")

    gridC = np.geomspace(C.min() / 2, C.max() * 4, 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        gridC,
        bC.point.predict(gridC),
        color=CATEGORICAL[1],
        label=f"fit: {bC.point.E:.2f} + B·C^-{bC.point.alpha:.3f}",
    )
    for c_, l_, col, n in zip(C, L, colors, N, strict=True):
        ax.plot([c_], [l_], "o", color=col, mec="white", mew=1.5, ms=9, zorder=3)
        ax.annotate(
            f"{n / 1e6:.2f}M",
            (c_, l_),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
            color=TEXT_2,
        )
    ax.set_xscale("log")
    ax.set_xlabel("training compute C = 6·N·D (FLOPs)")
    ax.set_ylabel("final val loss (nats/token)")
    ax.set_title("Loss vs compute")
    ax.legend(fontsize=8.5)
    save(fig, out_dir / "loss_vs_compute.png")

    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.axhline(0, color=TEXT_2, lw=1)
    ax.errorbar(N, resid, yerr=1.96 * np.array(sem), fmt="o", color=CATEGORICAL[0], capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("non-embedding parameters N")
    ax.set_ylabel("residual (nats)")
    ax.set_title("Fit residuals (bars: ±1.96 SEM of eval noise)")
    save(fig, out_dir / "scaling_residuals.png")

    multi_curves_figure(runs, out_dir / "scaling_curves.png")

    result = {
        "N": N.tolist(),
        "D": [r["tokens"] for r in runs],
        "C_6ND": C.tolist(),
        "val_loss": L.tolist(),
        "val_loss_sem": sem,
        "lr": [r["lr"] for r in runs],
        "fit_N": {
            "E": bN.point.E,
            "A": bN.point.A,
            "alpha": bN.point.alpha,
            "ci95": bN.ci,
            "n_boot_ok": len(bN.samples),
        },
        "fit_C": {
            "E": bC.point.E,
            "B": bC.point.A,
            "alpha": bC.point.alpha,
            "ci95": bC.ci,
            "n_boot_ok": len(bC.samples),
        },
        "residuals": resid.tolist(),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "leave_one_out": loo,
        "extrapolation_test": extrap,
    }
    return result
