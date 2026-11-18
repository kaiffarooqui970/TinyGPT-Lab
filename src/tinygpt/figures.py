"""All report figures, each built only from logged results (run dirs, results JSON/NPZ).

`scripts/make_figures.py` calls these; nothing here trains or evaluates models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from tinygpt.data.prepare import resolve
from tinygpt.scaling.analyze import (
    loss_curves_figure,
    lr_sweep_figure,
    scaling_analysis,
    sweep_runs,
)
from tinygpt.utils.plotting import CATEGORICAL, TEXT_2, apply_style, ordinal_blues, save


@dataclass
class FiguresConfig:
    out: str = "report"
    lr_sweep: str | None = "lr_sweep_s1"
    scaling_sweep: str | None = "scaling"
    lrcheck_sweep: str | None = "scaling_lrcheck"
    main_model_prefix: str = "s5"  # Phase 4 model = this size in the scaling sweep
    acts_dir: str | None = "data/acts/s5_l3"
    sae_sweep: str | None = "sweep_s5"
    n_boot: int = 1000


# ---------------------------------------------------------------------------- SAE
def sae_runs(sweep: str) -> list[dict[str, Any]]:
    root = resolve("runs") / "sae" / sweep
    latest: dict[str, Path] = {}
    for d in sorted(root.glob("*_*")):
        if (d / "final.json").exists():
            latest[d.name.split("_", 1)[1]] = d
    return [
        {"name": n, "dir": str(d), **json.loads((d / "final.json").read_text())}
        for n, d in latest.items()
    ]


def sae_pareto_figure(sweep: str, out_png: str | Path) -> list[dict[str, Any]]:
    """Sparsity (L0) vs fidelity (CE increase and FVU) for TopK and L1 SAEs."""
    apply_style()
    runs = sae_runs(sweep)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for kind, c, mk in [("topk", CATEGORICAL[0], "o"), ("l1", CATEGORICAL[1], "s")]:
        rs = sorted((r for r in runs if r["kind"] == kind), key=lambda r: r["l0"])
        if not rs:
            continue
        l0 = [r["l0"] for r in rs]
        axes[0].plot(l0, [r["ce_delta"] for r in rs], "-" + mk, color=c, label=kind.upper())
        axes[1].plot(l0, [1 - r["fve"] for r in rs], "-" + mk, color=c, label=kind.upper())
        for r in rs:
            tag = f"k={r['k']}" if kind == "topk" else f"λ={r['l1_coef']:g}"
            axes[0].annotate(
                tag,
                (r["l0"], r["ce_delta"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7.5,
                color=TEXT_2,
            )
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("L0 (active latents per token)")
        ax.legend()
    axes[0].set_ylabel("CE increase when spliced (nats)")
    axes[0].set_title("Sparsity vs downstream fidelity")
    axes[1].set_ylabel("fraction of variance unexplained")
    axes[1].set_title("Sparsity vs reconstruction")
    save(fig, out_png)
    return runs


def activation_norm_figure(acts_dir: str, out_png: str | Path) -> dict[str, Any]:
    apply_style()
    meta = json.loads((resolve(acts_dir) / "meta.json").read_text())
    pos = np.array(meta["norm_by_position"])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(len(pos)), pos, color=CATEGORICAL[0])
    ax.set_xlabel("position in window")
    ax.set_ylabel("mean ‖residual‖")
    ax.set_title(f"Residual norm after layer {meta['layer']} by position")
    save(fig, out_png)
    return {
        k: meta[k]
        for k in (
            "n_tokens",
            "d_model",
            "layer",
            "norm_mean",
            "norm_std",
            "norm_max",
            "mean_vector_norm",
        )
    }


# ---------------------------------------------------------------------------- Phase 8
def interventions_figure(results_json: str | Path, out_png: str | Path) -> None:
    apply_style()
    results = json.loads(Path(results_json).read_text())
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for i, n in enumerate(names):
        r = results[n]["ablation"]
        m, lo, hi = r["delta_watch_prob"]
        ax.errorbar(
            [i],
            [m],
            yerr=[[m - lo], [hi - m]],
            fmt="o",
            color=CATEGORICAL[0],
            capsize=4,
            ms=8,
            label="target feature (95% CI)" if i == 0 else None,
        )
        ctrl = [c for c in r["controls_delta_watch_prob"] if np.isfinite(c)]
        ax.plot(
            [i + 0.2] * len(ctrl),
            ctrl,
            "o",
            ms=4,
            color=CATEGORICAL[1],
            alpha=0.75,
            label="frequency-matched random features" if i == 0 else None,
        )
    ax.axhline(0, color=TEXT_2, lw=1)
    ax.set_xticks(range(len(names)), names)
    ax.set_xlim(-0.5, len(names) - 0.3)
    ax.set_ylabel("Δ P(watch tokens), positions where it fires")
    ax.set_title("(a) Ablating the feature")
    ax.legend(fontsize=8)
    ax = axes[1]
    for i, n in enumerate(names):
        st = results[n]["steering"]
        mults = sorted(st, key=float)
        ys = [st[mults[0]]["watch_rate"]["baseline_mean"]] + [
            st[m]["watch_rate"]["steered_mean"] for m in mults
        ]
        ax.plot([0.0] + [float(m) for m in mults], ys, "-o", color=CATEGORICAL[i % 8], label=n)
    ax.set_xlabel("steering strength (× max activation / 4)")
    ax.set_ylabel("watch-token rate in generated text")
    ax.set_title("(b) Steering along the decoder direction")
    ax.legend(fontsize=8)
    save(fig, out_png)


# ---------------------------------------------------------------------------- Phase 9
def universality_figures(results_dir: str | Path, fig_dir: str | Path) -> None:
    apply_style()
    results_dir, fig_dir = Path(results_dir), Path(fig_dir)
    A = np.load(results_dir / "universality_arrays.npz")
    res = json.loads((results_dir / "universality.json").read_text())
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bins = np.linspace(0, 1, 41)
    labels = [
        ("maxcorr_s5b_s5", "S5 vs S5, other SAE seed (ceiling)"),
        ("maxcorr_s4_s5", "S5 latents matched in S4"),
        ("maxcorr_s3_s5", "S5 latents matched in S3"),
    ]
    for c, (key, lab) in zip(CATEGORICAL, labels, strict=False):
        if key in A:
            v = A[key][np.isfinite(A[key])]
            ax.hist(
                v,
                bins=bins,
                histtype="step",
                lw=2,
                color=c,
                density=True,
                label=f"{lab} — median {np.median(v):.2f}",
            )
    ax.set_xlabel("best activation correlation with any latent of the other SAE")
    ax.set_ylabel("density over S5 latents")
    ax.set_title("Are S5's features present in smaller models?")
    ax.legend(fontsize=8)
    save(fig, fig_dir / "universality_hist.png")

    if "maxcorr_s4_s5" in A and "maxcorr_s3_s5" in A:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        freq = A["freq_s5"]
        for ax, key, lab, c in [
            (axes[0], "maxcorr_s4_s5", "S4", CATEGORICAL[1]),
            (axes[1], "maxcorr_s3_s5", "S3", CATEGORICAL[2]),
        ]:
            v = A[key]
            ok = np.isfinite(v) & (freq > 0)
            ax.scatter(freq[ok], v[ok], s=4, alpha=0.3, color=c, lw=0)
            # Binned median, so the trend is readable through the cloud.
            edges = np.geomspace(freq[ok].min(), freq[ok].max(), 12)
            mids, meds = [], []
            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                m = ok & (freq >= lo) & (freq < hi)
                if m.sum() >= 10:
                    mids.append(np.sqrt(lo * hi))
                    meds.append(np.median(v[m]))
            ax.plot(mids, meds, "-o", color=TEXT_2, ms=4, label="binned median")
            ax.set_xscale("log")
            ax.set_xlabel("S5 latent firing frequency")
            ax.set_ylabel(f"best match correlation in {lab}")
            ax.legend(fontsize=8)
        axes[0].set_title("Matching vs feature frequency")
        save(fig, fig_dir / "universality_vs_freq.png")

    q = res["quality"]
    names = [n for n in ("s3", "s4", "s5") if n in q]
    if names:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for ax, key, lab in zip(
            axes,
            ["fve", "ce_recovered", "dead_frac"],
            ["variance explained", "CE loss recovered", "dead latent fraction"],
            strict=True,
        ):
            ax.plot(
                [q[n]["d_model"] for n in names],
                [q[n][key] for n in names],
                "-o",
                color=CATEGORICAL[0],
            )
            for n in names:
                ax.annotate(
                    n.upper(),
                    (q[n]["d_model"], q[n][key]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                    color=TEXT_2,
                )
            ax.set_xlabel("model width d_model")
            ax.set_title(lab)
        save(fig, fig_dir / "sae_quality_vs_size.png")


def lrcheck_table(main: str, check: str) -> list[dict[str, Any]]:
    rows = [
        r
        for r in sweep_runs(main) + sweep_runs(check)
        if not r.get("diverged") and r["name"].startswith("s3")
    ]
    return sorted(
        ({"run": r["name"], "lr": r["lr"], "val_loss": r["val_loss"]} for r in rows),
        key=lambda r: r["lr"],
    )


def make_all(cfg: FiguresConfig) -> dict[str, Any]:
    """Regenerate every figure that has data available; return the key numbers."""
    out = resolve(cfg.out)
    figs, res_dir = out / "figures", out / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    if cfg.lr_sweep:
        summary["lr_sweep"] = lr_sweep_figure(cfg.lr_sweep, figs / "lr_sweep.png")
    if cfg.scaling_sweep:
        runs = sweep_runs(cfg.scaling_sweep)
        main = [r for r in runs if r["name"].startswith(cfg.main_model_prefix + "_")]
        if main:
            summary["main_model"] = loss_curves_figure(
                main[0]["dir"],
                figs / "main_model_loss.png",
                f"{cfg.main_model_prefix.upper()}: N = {main[0]['n_params_nonembed'] / 1e6:.1f}M"
                " non-embedding params",
            )
        if len({r["n_params_nonembed"] for r in runs}) >= 3:
            fit = scaling_analysis(cfg.scaling_sweep, figs, n_boot=cfg.n_boot)
            (res_dir / "scaling_fit.json").write_text(json.dumps(fit, indent=2))
            summary["scaling_fit"] = {k: fit[k] for k in ("fit_N", "rmse", "extrapolation_test")}
    if cfg.lrcheck_sweep and cfg.scaling_sweep:
        summary["lr_check_s3"] = lrcheck_table(cfg.scaling_sweep, cfg.lrcheck_sweep)
    if cfg.acts_dir and (resolve(cfg.acts_dir) / "meta.json").exists():
        summary["activations"] = activation_norm_figure(cfg.acts_dir, figs / "activation_norms.png")
    if cfg.sae_sweep and sae_runs(cfg.sae_sweep):
        runs = sae_pareto_figure(cfg.sae_sweep, figs / "sae_pareto.png")
        summary["sae_sweep"] = [
            {
                k: r[k]
                for k in (
                    "name",
                    "kind",
                    "k",
                    "l1_coef",
                    "l0",
                    "fve",
                    "ce_delta",
                    "ce_recovered",
                    "dead_frac",
                )
            }
            for r in runs
        ]
    if (res_dir / "interventions.json").exists():
        interventions_figure(res_dir / "interventions.json", figs / "interventions.png")
    if (res_dir / "universality.json").exists():
        universality_figures(res_dir, figs)
    (res_dir / "figures_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


__all__ = ["FiguresConfig", "make_all", "ordinal_blues"]
