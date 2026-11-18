"""AdamW parameter groups and the warmup + cosine learning-rate schedule."""

from __future__ import annotations

import math

import torch
from torch import nn


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into decay / no-decay groups.

    Decay: weight matrices of Linear layers. No decay: norm gains and embeddings. The
    tied unembedding *is* the embedding, so it is excluded too. Decaying a norm gain pulls
    it toward 0, which fights the normalisation itself. Embeddings are sparse lookups, so
    decay on them mostly shrinks rare tokens. (Common practice from GPT-2/3 and minGPT.)
    """
    decay, no_decay, seen = [], [], set()
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            if (
                isinstance(module, nn.Linear)
                and name == "weight"
                and not _is_tied_embedding(model, p)
            ):
                decay.append(p)
            else:
                no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _is_tied_embedding(model: nn.Module, p: torch.Tensor) -> bool:
    emb = getattr(model, "tok_emb", None)
    return emb is not None and p is emb.weight


def make_optimizer(
    model: nn.Module, lr: float, weight_decay: float, betas: tuple[float, float], eps: float = 1e-8
) -> torch.optim.AdamW:
    """AdamW with decoupled weight decay (Loshchilov & Hutter, 2019): the decay is applied
    directly to the weights, not added to the gradient, so Adam's per-parameter scaling
    does not rescale it."""
    return torch.optim.AdamW(param_groups(model, weight_decay), lr=lr, betas=betas, eps=eps)


def lr_at(step: int, max_lr: float, warmup: int, total: int, min_lr_ratio: float = 0.1) -> float:
    """Linear warmup over `warmup` steps, then cosine decay to `min_lr_ratio * max_lr` at
    `total`. The schedule is a pure function of the step, so resume needs no scheduler state."""
    if warmup > 0 and step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return max_lr * min_lr_ratio
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)
