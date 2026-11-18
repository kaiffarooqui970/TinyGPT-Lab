"""Causal tests of SAE features: ablation (remove a feature's contribution) and steering
(add its decoder direction during generation).

Both edit the residual stream after `layer` blocks, which is where the SAE was trained:
* ablate:  r <- r - z_i(r) * d_i      (d_i = decoder column in raw activation units)
* steer:   r <- r + alpha * d_i       (at every position, including during generation)
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tinygpt.model.generate import generate
from tinygpt.model.gpt import GPT
from tinygpt.sae.sae import SAE


def decoder_dir(sae: SAE, i: int) -> torch.Tensor:
    """Feature i's decoder direction in raw activation units. Adding z * d_raw to the raw
    residual equals adding z * d to the normalised one."""
    return sae.W_dec[:, i].detach() / sae.scale


def ablate_fn(sae: SAE, feats: Sequence[int]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Patch that subtracts the given features' contributions (z_i * d_i) from the residual.
    Only the feature is removed: the SAE's reconstruction error stays in the residual."""
    idx = torch.tensor(list(feats))

    def fn(r: torch.Tensor) -> torch.Tensor:
        z = sae.encode(r)[..., idx.to(r.device)]  # (B, T, F)
        D = (sae.W_dec[:, idx.to(r.device)] / sae.scale).T  # (F, D)
        return r - (z @ D).to(r.dtype)

    return fn


@torch.no_grad()
def ablation_effect(
    model: GPT,
    sae: SAE,
    feature: int,
    layer: int,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    watch_tokens: Sequence[int] = (),
) -> dict[str, Any]:
    """Ablate `feature` and measure, *on the positions where it fires*:
    the change in next-token loss, the KL(clean || ablated), and the change in total
    probability of `watch_tokens`. Returns per-position arrays for bootstrap CIs."""
    d_loss, kls, d_watch = [], [], []
    patch = ablate_fn(sae, [feature])
    watch = torch.tensor(list(watch_tokens), device=device) if watch_tokens else None
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        r = model.residual(x, layer)
        active = sae.encode(r)[..., feature] > 0  # (B, T)
        if not bool(active.any()):
            continue
        clean, _, _ = model(x)
        abl, _ = model.forward_patched(x, layer, patch)
        lp_c = F.log_softmax(clean.float(), -1)[active]
        lp_a = F.log_softmax(abl.float(), -1)[active]
        tgt = y[active]
        d_loss.append((lp_c.gather(-1, tgt[:, None]) - lp_a.gather(-1, tgt[:, None]))[:, 0])
        kls.append((lp_c.exp() * (lp_c - lp_a)).sum(-1))
        if watch is not None:
            d_watch.append(lp_a.exp()[:, watch].sum(-1) - lp_c.exp()[:, watch].sum(-1))
    cat = lambda xs: torch.cat(xs).cpu().numpy() if xs else np.zeros(0)  # noqa: E731
    return {"delta_loss": cat(d_loss), "kl": cat(kls), "delta_watch_prob": cat(d_watch)}


@contextlib.contextmanager
def steering(model: GPT, layer: int, vec: torch.Tensor) -> Iterator[None]:
    """Add `vec` to the residual after `layer` blocks (layer >= 1) at every position, for
    any forward pass in the block, including cached generation."""
    if layer < 1:
        raise ValueError("steering hooks need layer >= 1")

    def hook(_m: torch.nn.Module, _inp: tuple, out: tuple) -> tuple:
        x, kv = out
        return (x + vec.to(x.dtype), kv)

    h = model.blocks[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@torch.no_grad()
def steered_generations(
    model: GPT,
    prompts: Sequence[list[int]],
    layer: int,
    vec: torch.Tensor | None,
    n_per_prompt: int,
    max_new: int,
    seed: int,
    device: torch.device,
    eos_id: int | None = None,
) -> list[list[int]]:
    """Sample continuations (temperature 0.8, top-k 50) with or without steering. The
    same seeds are used in both conditions, so differences come from the intervention."""
    outs = []
    ctx = steering(model, layer, vec) if vec is not None else contextlib.nullcontext()
    with ctx:
        for p_i, p in enumerate(prompts):
            for s in range(n_per_prompt):
                g = torch.Generator().manual_seed(seed + 1000 * p_i + s)
                ids = torch.tensor([p], device=device)
                out = generate(
                    model, ids, max_new, temperature=0.8, top_k=50, generator=g, eos_id=eos_id
                )
                outs.append(out[0, len(p) :].tolist())
    return outs


def bootstrap_mean_ci(
    x: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, (n_boot, x.size))].mean(1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(x.mean()), float(lo), float(hi)


def bootstrap_diff_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Difference of means (a - b) with a bootstrap CI, and Cohen's d (pooled SD)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    diffs = a[rng.integers(0, a.size, (n_boot, a.size))].mean(1) - b[
        rng.integers(0, b.size, (n_boot, b.size))
    ].mean(1)
    pooled = np.sqrt(
        ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / max(1, a.size + b.size - 2)
    )
    return {
        "diff": float(a.mean() - b.mean()),
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
        "cohens_d": float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan"),
    }
