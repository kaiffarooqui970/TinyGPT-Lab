"""SAE evaluation: reconstruction metrics on held-out activations and downstream CE when
the reconstruction is spliced into the model."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from tinygpt.model.gpt import GPT
from tinygpt.sae.sae import SAE


@torch.no_grad()
def activation_metrics(
    sae: SAE, acts: torch.Tensor, batch_size: int = 8192, device: str | torch.device = "cpu"
) -> dict[str, float]:
    """MSE, fraction of variance explained, L0, and latent firing statistics over `acts`.

    FVE uses the variance of the *whole* held-out set (two-pass), not per batch:
    FVE = 1 - sum ||x - x_hat||^2 / sum ||x - mean||^2, in normalised units.
    """
    sae = sae.to(device).eval()
    n = acts.shape[0]
    mean = torch.zeros(sae.cfg.d_in, dtype=torch.float64, device=device)
    for i in range(0, n, batch_size):
        mean += (acts[i : i + batch_size].to(device).float() * sae.scale).sum(0).double()
    mean /= n
    sse = sst = l0 = 0.0
    fires = torch.zeros(sae.n_latents, dtype=torch.float64, device=device)
    for i in range(0, n, batch_size):
        x = acts[i : i + batch_size].to(device).float() * sae.scale
        z = sae.activate(sae.pre_acts(x))
        x_hat = sae.decode(z)
        sse += float((x - x_hat).pow(2).sum())
        sst += float((x.double() - mean).pow(2).sum())
        active = z > 0
        l0 += float(active.sum())
        fires += active.sum(0).double()
    freq = fires / n
    return {
        "mse": sse / n,
        "fve": 1.0 - sse / sst,
        "l0": l0 / n,
        "dead_frac": float((fires == 0).float().mean()),  # never fired on the held-out set
        "dense_frac": float((freq > 0.1).float().mean()),  # fire on more than 10% of tokens
        "n_tokens": n,
    }


@torch.no_grad()
def downstream_ce(
    model: GPT,
    sae: SAE | None,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    layer: int,
    device: str | torch.device,
) -> dict[str, float]:
    """CE loss of the model when the residual after `layer` blocks is (a) left alone,
    (b) replaced by the SAE reconstruction, (c) zero-ablated, (d) mean-ablated.

    `ce_recovered` = (CE_zero - CE_sae) / (CE_zero - CE_clean): 1 means the splice is
    lossless, 0 means it is as bad as deleting the layer's residual stream entirely.
    """
    model = model.to(device).eval()
    tot: dict[str, float] = {"clean": 0.0, "sae": 0.0, "zero": 0.0, "mean": 0.0}
    nb = 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        _, l_clean, _ = model(x, y)
        tot["clean"] += float(l_clean)
        if sae is not None:
            sae = sae.to(device)
            _, l_sae = model.forward_patched(x, layer, sae.reconstruct, y)
            tot["sae"] += float(l_sae)
        _, l_zero = model.forward_patched(x, layer, torch.zeros_like, y)
        tot["zero"] += float(l_zero)
        _, l_mean = model.forward_patched(
            x, layer, lambda r: r.mean(dim=(0, 1), keepdim=True).expand_as(r), y
        )
        tot["mean"] += float(l_mean)
        nb += 1
    out = {f"ce_{k}": v / nb for k, v in tot.items()}
    if sae is not None:
        out["ce_delta"] = out["ce_sae"] - out["ce_clean"]
        out["ce_recovered"] = (out["ce_zero"] - out["ce_sae"]) / (out["ce_zero"] - out["ce_clean"])
    return out
