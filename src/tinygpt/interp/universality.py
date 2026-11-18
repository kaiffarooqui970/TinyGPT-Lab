"""Feature universality across models: match SAE latents by activation correlation.

Two SAEs trained on different models (or seeds) are compared on the *same tokens*. For
latents a (SAE A) and b (SAE B), let corr(a, b) be the Pearson correlation of their
activations over those tokens. For every latent b we report

    max_corr_B(b) = max_a corr(a, b)

which is high when SAE A has a latent tracking the same thing. This works across
different d_model, unlike decoder cosine similarity, which needs a shared space.
"""

from __future__ import annotations

import numpy as np
import torch

from tinygpt.sae.sae import SAE


@torch.no_grad()
def latent_matrix(
    sae: SAE, acts: torch.Tensor, batch_size: int = 8192, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Dense latent activations (n_tokens, n_latents) as float16, on the CPU."""
    sae = sae.to(device).eval()
    out = []
    for i in range(0, acts.shape[0], batch_size):
        out.append(sae.encode(acts[i : i + batch_size].to(device)).half().cpu())
    return torch.cat(out)


def _standardise(z: torch.Tensor, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    z = z.to(device).float()
    mu = z.mean(0)
    sd = z.std(0)
    alive = sd > 0
    return (z - mu) / sd.clamp_min(1e-8), alive


@torch.no_grad()
def max_correlation(
    zA: torch.Tensor, zB: torch.Tensor, chunk: int = 1024, device: str | torch.device = "cpu"
) -> dict[str, np.ndarray]:
    """For each latent of B: the max Pearson correlation with any latent of A, plus its
    argmax. Dead latents (zero variance) get NaN. The correlation matrix is computed in
    column chunks of B to bound memory."""
    n = zA.shape[0]
    if zB.shape[0] != n:
        raise ValueError("both latent matrices must cover the same tokens")
    sA, aliveA = _standardise(zA, device)
    sA = sA[:, aliveA]
    idxA = torch.nonzero(aliveA).squeeze(1)
    best = torch.full((zB.shape[1],), float("nan"))
    arg = torch.full((zB.shape[1],), -1, dtype=torch.long)
    for j in range(0, zB.shape[1], chunk):
        sB, aliveB = _standardise(zB[:, j : j + chunk], device)
        c = (sA.T @ sB) / (n - 1)  # (alive A, chunk)
        v, a = c.max(0)
        v, a = v.cpu(), idxA[a].cpu()
        v[~aliveB.cpu()] = float("nan")
        best[j : j + chunk] = v
        arg[j : j + chunk] = a
    return {"max_corr": best.numpy(), "argmax": arg.numpy()}


def summarise(
    max_corr: np.ndarray, thresholds: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)
) -> dict[str, float]:
    mc = max_corr[np.isfinite(max_corr)]
    out = {"n_alive": int(mc.size), "mean": float(mc.mean()), "median": float(np.median(mc))}
    for t in thresholds:
        out[f"frac_ge_{t}"] = float((mc >= t).mean())
    return out
