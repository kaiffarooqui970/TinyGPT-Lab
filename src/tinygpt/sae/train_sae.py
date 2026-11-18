"""Train a TopK or L1 SAE on collected activations, then evaluate it."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tinygpt.data.loader import TokenLoader
from tinygpt.data.prepare import resolve, shard_paths
from tinygpt.model.gpt import load_gpt
from tinygpt.sae.collect import ActivationStore, eval_min_start
from tinygpt.sae.eval_sae import activation_metrics, downstream_ce
from tinygpt.sae.sae import SAE, SAEConfig
from tinygpt.train.metrics import pick_device
from tinygpt.utils.logging import JSONLLogger
from tinygpt.utils.run import create_run_dir
from tinygpt.utils.seed import seed_everything


@dataclass
class SAETrainConfig:
    name: str = "sae"
    acts_dir: str = "data/acts/s5_l3"
    kind: str = "topk"
    expansion: int = 16
    k: int = 32
    l1_coef: float = 1e-3
    aux_k: int | None = None
    aux_coef: float = 1.0 / 32.0
    train_tokens: int = 8_000_000  # may exceed the store: then it does several epochs
    heldout_tokens: int = 262_144  # last tokens of the store, never trained on
    max_tokens: int | None = None  # use only the first N store tokens (equal budgets)
    batch_size: int = 4096
    lr: float = 5e-4
    decay_frac: float = 0.2  # linear LR decay to 0 over the final fraction
    l1_warmup_frac: float = 0.05  # L1 coefficient ramps up (avoids early dead latents)
    dead_tokens: int = 1_000_000
    log_interval: int = 100
    ce_eval_batches: int = 20
    ce_eval_batch_size: int = 32
    data_dir: str = "data/tinystories"
    device: str = "auto"
    seed: int = 0
    runs_root: str = "runs/sae"


def sae_lr_at(step: int, total: int, lr: float, decay_frac: float) -> float:
    start = int(total * (1 - decay_frac))
    if step < start:
        return lr
    return lr * max(0.0, (total - step) / max(1, total - start))


def train_sae(cfg: SAETrainConfig, run_dir: Path | None = None) -> tuple[SAE, dict[str, Any]]:
    """Train, evaluate and save an SAE. Returns the SAE and its final metrics."""
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    store = ActivationStore(cfg.acts_dir)
    acts, _ = store.load_all(max_tokens=cfg.max_tokens)
    n_train = len(acts) - cfg.heldout_tokens
    if n_train <= cfg.batch_size:
        raise ValueError("activation store too small for heldout_tokens + one batch")
    train_acts, held = acts[:n_train], acts[n_train:]
    meta = store.meta

    sae = SAE(
        SAEConfig(
            d_in=meta["d_model"],
            expansion=cfg.expansion,
            kind=cfg.kind,
            k=cfg.k,
            l1_coef=cfg.l1_coef,
            aux_k=cfg.aux_k,
            aux_coef=cfg.aux_coef,
        )
    )
    g = torch.Generator().manual_seed(cfg.seed)
    stats_idx = torch.randint(0, n_train, (min(n_train, 65536),), generator=g)
    sae.set_input_stats(train_acts[stats_idx].float())
    sae = sae.to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr, betas=(0.9, 0.999))

    run_dir = run_dir or create_run_dir(cfg, root=resolve(cfg.runs_root), name=cfg.name)
    log = JSONLLogger(run_dir / "metrics.jsonl")
    total = math.ceil(cfg.train_tokens / cfg.batch_size)
    since_fired = torch.zeros(sae.n_latents, dtype=torch.long, device=device)
    perm = torch.randperm(n_train, generator=g)
    pos = 0
    t0 = time.perf_counter()
    for step in range(total):
        if pos + cfg.batch_size > n_train:  # new epoch, new shuffle
            perm, pos = torch.randperm(n_train, generator=g), 0
        x = train_acts[perm[pos : pos + cfg.batch_size]].to(device).float()
        pos += cfg.batch_size
        lr = sae_lr_at(step, total, cfg.lr, cfg.decay_frac)
        for grp in opt.param_groups:
            grp["lr"] = lr
        if cfg.kind == "l1":
            ramp = min(1.0, (step + 1) / max(1, int(cfg.l1_warmup_frac * total)))
            sae.cfg.l1_coef = cfg.l1_coef * ramp
        dead = since_fired >= cfg.dead_tokens
        out = sae(x, dead_mask=dead)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        sae.project_decoder_grad()
        opt.step()
        sae.normalize_decoder()
        with torch.no_grad():
            fired = (out.z > 0).any(0)
            since_fired += cfg.batch_size
            since_fired[fired] = 0
        if step % cfg.log_interval == 0 or step == total - 1:
            log.log(
                {
                    "step": step,
                    "tokens": (step + 1) * cfg.batch_size,
                    "loss": out.loss.item(),
                    "mse": out.mse.item(),
                    "fvu": out.fvu.item(),
                    "aux_loss": out.aux_loss.item(),
                    "l0": float((out.z > 0).sum(-1).float().mean()),
                    "dead_frac": float(dead.float().mean()),
                    "lr": lr,
                    "elapsed_s": time.perf_counter() - t0,
                }
            )

    sae.cfg.l1_coef = cfg.l1_coef
    metrics = evaluate_sae(sae, held, cfg, meta, device)
    metrics["dead_frac_train"] = float((since_fired >= cfg.dead_tokens).float().mean())
    metrics.update(
        {
            "kind": cfg.kind,
            "k": cfg.k,
            "l1_coef": cfg.l1_coef,
            "expansion": cfg.expansion,
            "n_latents": sae.n_latents,
            "train_tokens": total * cfg.batch_size,
            "n_train_acts": n_train,
            "epochs": total * cfg.batch_size / n_train,
            "train_seconds": time.perf_counter() - t0,
        }
    )
    log.log({"final": True, **metrics})
    log.close()
    torch.save(
        {"state_dict": sae.state_dict(), "sae_cfg": sae.cfg.__dict__, "acts_meta": meta},
        run_dir / "sae.pt",
    )
    (run_dir / "final.json").write_text(json.dumps(metrics, indent=2))
    return sae, metrics


def evaluate_sae(
    sae: SAE, held: torch.Tensor, cfg: SAETrainConfig, meta: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    m = activation_metrics(sae, held, device=device)
    model, _ = load_gpt(resolve(meta["checkpoint"]), device)
    val = TokenLoader(
        shard_paths(cfg.data_dir, "val"),
        model.cfg.block_size,
        cfg.ce_eval_batch_size,
        seed=0,
        min_start=eval_min_start(meta),  # never evaluate on tokens the SAE trained on
    )
    m.update(
        downstream_ce(
            model, sae, val.fixed_batches(cfg.ce_eval_batches, seed=4321), meta["layer"], device
        )
    )
    return m


def load_sae(path: str | Path, device: str | torch.device = "cpu") -> tuple[SAE, dict]:
    ckpt = torch.load(resolve(path), map_location="cpu", weights_only=False)
    sae = SAE(SAEConfig(**ckpt["sae_cfg"]))
    sae.load_state_dict(ckpt["state_dict"])
    return sae.to(device).eval(), ckpt
