"""Sparse autoencoders on residual-stream activations: TopK (Gao et al. 2024) and L1
(Bricken et al. 2023).

    x_n   = x * scale                                   (input normalisation)
    pre   = W_enc (x_n - b_dec) + b_enc                 (pre-encoder bias b_dec)
    z     = TopK(pre) then ReLU  |  ReLU(pre)           (topk | l1)
    x_hat = W_dec z + b_dec,   ||W_dec[:, i]|| = 1

Decoder columns are kept at unit norm. Otherwise the L1 penalty (and, for TopK, the
choice of which latents win) can be gamed by shrinking z and growing the column. We
project the gradient's parallel component out before each step and renormalise after it.

Dead latents in TopK are handled by the AuxK loss: the top `aux_k` *dead* latents
(those that have not fired for `dead_tokens` tokens) try to reconstruct the residual
x_n - x_hat. That gives them gradient again (Gao et al. 2024, sec. 2.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SAEConfig:
    d_in: int
    expansion: int = 16
    kind: str = "topk"  # "topk" | "l1"
    k: int = 32
    l1_coef: float = 1e-3
    aux_k: int | None = None  # default: d_in // 2
    aux_coef: float = 1.0 / 32.0


@dataclass
class SAEOutput:
    x_hat: torch.Tensor  # normalised units
    z: torch.Tensor
    loss: torch.Tensor
    mse: torch.Tensor
    fvu: torch.Tensor  # fraction of variance unexplained (batch estimate)
    aux_loss: torch.Tensor
    l1: torch.Tensor | None


class SAE(nn.Module):
    """TopK or L1 sparse autoencoder."""

    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()
        if cfg.kind not in ("topk", "l1"):
            raise ValueError(f"unknown SAE kind {cfg.kind!r}")
        self.cfg = cfg
        d, n = cfg.d_in, cfg.d_in * cfg.expansion
        self.n_latents = n
        W = torch.randn(d, n)
        W = W / W.norm(dim=0, keepdim=True)
        self.W_dec = nn.Parameter(W)
        # Tied init W_enc = W_dec^T: at init each latent reads the direction it writes, which
        # cuts dead latents early on (Gao et al. 2024).
        self.W_enc = nn.Parameter(W.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(n))
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.register_buffer("scale", torch.tensor(1.0))

    # ------------------------------------------------------------------ data stats
    @torch.no_grad()
    def set_input_stats(self, x: torch.Tensor) -> None:
        """Set the normalisation scale so E||x*scale||^2 = d_in, and initialise b_dec to the
        mean of the normalised data (the pre-encoder bias then centres inputs from step 0)."""
        x = x.float()
        self.scale.copy_((self.cfg.d_in / x.pow(2).sum(-1).mean()).sqrt())
        self.b_dec.copy_((x * self.scale).mean(0))

    # ------------------------------------------------------------------ forward
    def pre_acts(self, x_n: torch.Tensor) -> torch.Tensor:
        return (x_n - self.b_dec) @ self.W_enc.T + self.b_enc

    def activate(self, pre: torch.Tensor) -> torch.Tensor:
        if self.cfg.kind == "l1":
            return F.relu(pre)
        vals, idx = pre.topk(self.cfg.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, F.relu(vals))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Latent codes for raw (un-normalised) inputs."""
        return self.activate(self.pre_acts(x.float() * self.scale))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruction in *normalised* units."""
        return z @ self.W_dec.T + self.b_dec

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction in the original activation units (for splicing into the model)."""
        return (self.decode(self.encode(x)) / self.scale).to(x.dtype)

    def forward(self, x: torch.Tensor, dead_mask: torch.Tensor | None = None) -> SAEOutput:
        x_n = x.float() * self.scale
        pre = self.pre_acts(x_n)
        z = self.activate(pre)
        x_hat = self.decode(z)
        err = x_n - x_hat
        mse = err.pow(2).sum(-1).mean()
        var = (x_n - x_n.mean(0)).pow(2).sum(-1).mean()
        fvu = mse / var.clamp_min(1e-8)
        loss = mse
        l1 = None
        if self.cfg.kind == "l1":
            l1 = z.sum(-1).mean()  # decoder columns have unit norm, so this is sum |z_i| ||d_i||
            loss = loss + self.cfg.l1_coef * l1
        aux = torch.zeros((), device=x.device)
        if self.cfg.kind == "topk" and dead_mask is not None and bool(dead_mask.any()):
            aux = self._aux_loss(pre, err, dead_mask)
            loss = loss + self.cfg.aux_coef * aux
        return SAEOutput(x_hat, z, loss, mse, fvu, aux, l1)

    def _aux_loss(
        self, pre: torch.Tensor, err: torch.Tensor, dead_mask: torch.Tensor
    ) -> torch.Tensor:
        k_aux = min(self.cfg.aux_k or self.cfg.d_in // 2, int(dead_mask.sum()))
        masked = pre.masked_fill(~dead_mask, float("-inf"))
        vals, idx = masked.topk(k_aux, dim=-1)
        z_aux = torch.zeros_like(pre).scatter_(-1, idx, F.relu(vals))
        err_hat = z_aux @ self.W_dec.T  # no bias: predicts the residual
        target = err.detach()
        # Normalised by the residual's variance, so the aux term's scale is independent of
        # how well the main reconstruction already does (as in Gao et al.).
        denom = (target - target.mean(0)).pow(2).sum(-1).mean().clamp_min(1e-8)
        return (target - err_hat).pow(2).sum(-1).mean() / denom

    # ------------------------------------------------------------------ constraints
    @torch.no_grad()
    def project_decoder_grad(self) -> None:
        """Remove the component of each column's gradient parallel to the column, so that
        a step moves along the unit sphere (to first order)."""
        if self.W_dec.grad is None:
            return
        W, G = self.W_dec, self.W_dec.grad
        G -= (G * W).sum(0, keepdim=True) * W

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec /= self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
