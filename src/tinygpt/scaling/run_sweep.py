"""Config-driven sweeps: a grid of model sizes x learning rates on top of a base config.

Each run goes to `runs/sweeps/<sweep>/<timestamp>_<run>/` and is skipped if a completed
run with the same name (one with a `final.json`) already exists, so an interrupted sweep
can be restarted with the same command.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tinygpt.data.prepare import resolve
from tinygpt.model.gpt import GPT, GPTConfig
from tinygpt.train.trainer import TrainConfig, Trainer
from tinygpt.utils.config import apply_overrides, from_dict
from tinygpt.utils.logging import JSONLLogger


@dataclass
class SweepConfig:
    """A sweep over model sizes and learning rates."""

    name: str
    base_config: str
    sizes: dict[str, dict[str, Any]]  # size name -> GPTConfig field overrides
    lrs: list[float] | None = None  # explicit grid of LRs (applied to every size)
    lr_rule: str | None = None  # "kaplan": transfer ref_lr from ref_size by Kaplan's LR(N)
    ref_lr: float | None = None
    ref_size: str | None = None
    lr_multipliers: list[float] = field(default_factory=lambda: [1.0])  # around the rule
    overrides: list[str] = field(default_factory=list)  # applied to every run
    size_overrides: dict[str, list[str]] = field(default_factory=dict)  # per-size extras


def kaplan_lr(n_params: float) -> float:
    """Kaplan et al. (2020), eq. D.1 fit of the best LR vs non-embedding params:
    LR(N) ~ 0.003239 - 0.0001395 * ln(N). Only the *ratio* between sizes is used here."""
    return 0.003239 - 0.0001395 * math.log(n_params)


def non_embed_params(model_overrides: dict[str, Any], base_model: dict[str, Any]) -> int:
    cfg = from_dict(GPTConfig, {**base_model, **model_overrides})
    return GPT(cfg).num_params(non_embedding=True)


def plan_runs(sweep: SweepConfig) -> list[tuple[str, list[str]]]:
    """Expand the sweep into (run_name, overrides) pairs."""
    base = yaml.safe_load(resolve(sweep.base_config).read_text())
    base_model = base.get("model", {})
    runs = []
    for size, model_kw in sweep.sizes.items():
        model_sets = [f"model.{k}={v}" for k, v in model_kw.items()]
        if sweep.lr_rule == "kaplan":
            assert sweep.ref_lr is not None and sweep.ref_size is not None
            n = non_embed_params(model_kw, base_model)
            n_ref = non_embed_params(sweep.sizes[sweep.ref_size], base_model)
            centre = sweep.ref_lr * kaplan_lr(n) / kaplan_lr(n_ref)
            lrs = [centre * m for m in sweep.lr_multipliers]
        elif sweep.lrs is not None:
            lrs = list(sweep.lrs)
        else:
            lrs = [float(base.get("lr", 1e-3))]
        for lr in lrs:
            name = f"{size}_lr{lr:.2e}"
            sets = [
                *sweep.overrides,
                *model_sets,
                *sweep.size_overrides.get(size, []),
                f"lr={lr}",
                f"name={name}",
            ]
            runs.append((name, sets))
    return runs


def completed_run(sweep_dir: Path, name: str) -> Path | None:
    for d in sorted(sweep_dir.glob(f"*_{name}")):
        if (d / "final.json").exists():
            return d
    return None


def run_sweep(sweep: SweepConfig, only: list[str] | None = None) -> list[dict[str, Any]]:
    """Run (or resume) every run in the sweep; return the summary rows."""
    base = yaml.safe_load(resolve(sweep.base_config).read_text())
    sweep_dir = resolve("runs") / "sweeps" / sweep.name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, sets in plan_runs(sweep):
        if only and not any(name.startswith(o) for o in only):
            continue
        done = completed_run(sweep_dir, name)
        if done is None:
            cfg = from_dict(TrainConfig, apply_overrides(json.loads(json.dumps(base)), sets))
            cfg.runs_root = str(sweep_dir)
            trainer = Trainer(cfg)
            print(
                f"[{sweep.name}] {name}: N={trainer.n_params:,} steps={trainer.total_steps}",
                flush=True,
            )
            try:
                trainer.train()
            except FloatingPointError as e:  # diverged: record it, keep the sweep going
                (trainer.run_dir / "final.json").write_text(
                    json.dumps(
                        {"diverged": True, "error": str(e), "n_params_nonembed": trainer.n_params}
                    )
                )
            trainer.close()
            done = trainer.run_dir
        res = json.loads((done / "final.json").read_text())
        row = {
            "run": name,
            "dir": str(done),
            "lr": _lr_of(sets),
            **{k: v for k, v in res.items() if k != "val_loss_per_batch"},
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    with JSONLLogger(sweep_dir / "summary.jsonl") as log:
        for r in rows:
            log.log(r)
    return rows


def _lr_of(sets: list[str]) -> float:
    return float(next(s for s in reversed(sets) if s.startswith("lr=")).split("=", 1)[1])
