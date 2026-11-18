"""Per-latent statistics: firing frequency, top-activating contexts, activation
histograms, and the logit effect of each decoder direction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tinygpt.model.gpt import GPT
from tinygpt.sae.sae import SAE

HIST_EDGES = np.linspace(-2.0, 2.0, 41)  # log10(activation) bins


@dataclass
class FeatureStats:
    freq: np.ndarray  # (n_latents,) fraction of tokens with z > 0
    mean_active: np.ndarray  # (n_latents,) mean activation when active
    top_vals: np.ndarray  # (n_top, n_latents), descending
    top_rows: np.ndarray  # (n_top, n_latents) global row index into the store
    hist: np.ndarray  # (n_latents, len(HIST_EDGES) - 1) counts of log10(z) for z > 0
    n_tokens: int


@torch.no_grad()
def compute_feature_stats(
    sae: SAE,
    acts: torch.Tensor,
    n_top: int = 20,
    batch_size: int = 8192,
    device: str | torch.device = "cpu",
) -> FeatureStats:
    """One pass over `acts` keeping a running top-`n_top` per latent."""
    sae = sae.to(device).eval()
    n_lat = sae.n_latents
    nb = len(HIST_EDGES) - 1
    top_v = torch.full((n_top, n_lat), -1.0, device=device)
    top_r = torch.zeros((n_top, n_lat), dtype=torch.long, device=device)
    # float64 accumulators on the CPU (MPS has no float64); per-batch reductions are fp32.
    fires = torch.zeros(n_lat, dtype=torch.float64)
    sums = torch.zeros(n_lat, dtype=torch.float64)
    hist = torch.zeros(n_lat * nb, dtype=torch.float32, device=device)
    edges = torch.tensor(HIST_EDGES[1:-1], dtype=torch.float32, device=device)
    for i in range(0, acts.shape[0], batch_size):
        z = sae.encode(acts[i : i + batch_size].to(device))
        active = z > 0
        fires += active.sum(0).cpu().double()
        sums += z.sum(0).cpu().double()
        k = min(n_top, z.shape[0])
        v, r = z.topk(k, dim=0)
        cat_v = torch.cat([top_v, v])
        cat_r = torch.cat([top_r, r + i])
        top_v, sel = cat_v.topk(n_top, dim=0)
        top_r = cat_r.gather(0, sel)
        rows, cols = active.nonzero(as_tuple=True)
        b = torch.bucketize(torch.log10(z[rows, cols]), edges)
        hist.index_add_(0, cols * nb + b, torch.ones_like(b, dtype=torch.float32))
    freq = (fires / acts.shape[0]).cpu().numpy()
    mean_active = (sums / fires.clamp_min(1)).cpu().numpy()
    return FeatureStats(
        freq=freq,
        mean_active=mean_active,
        top_vals=top_v.cpu().numpy(),
        top_rows=top_r.cpu().numpy(),
        hist=hist.view(n_lat, nb).cpu().double().numpy(),
        n_tokens=int(acts.shape[0]),
    )


@torch.no_grad()
def logit_effects(
    model: GPT, sae: SAE, latents: list[int], top: int = 10
) -> dict[int, dict[str, list[tuple[int, float]]]]:
    """Direct effect of each decoder direction on the logits: W_U (g * d_i), where g is
    the final RMSNorm gain. This ignores the later layers and the norm's rescaling, so it
    is the 'direct path' approximation used in feature dashboards."""
    W_U = model.lm_head.weight.float().cpu()  # (V, D)
    g = model.norm_f.weight.float().cpu()
    D = sae.W_dec.detach().float().cpu()  # (D, n_latents)
    out = {}
    for i in latents:
        eff = W_U @ (g * D[:, i])
        eff = eff - eff.mean()
        hi = torch.topk(eff, top)
        lo = torch.topk(-eff, top)
        out[i] = {
            "boost": list(zip(hi.indices.tolist(), hi.values.tolist(), strict=True)),
            "suppress": list(zip(lo.indices.tolist(), (-lo.values).tolist(), strict=True)),
        }
    return out


def context_of(
    row: int, tokens: torch.Tensor, block_size: int, before: int = 24, after: int = 4
) -> tuple[list[int], int, int]:
    """Token ids around global row `row`, clipped to its window.
    Returns (ids, start_row, index of the target token within ids)."""
    w0 = (row // block_size) * block_size
    start = max(w0, row - before)
    end = min(w0 + block_size, row + after + 1)
    return tokens[start:end].tolist(), start, row - start
