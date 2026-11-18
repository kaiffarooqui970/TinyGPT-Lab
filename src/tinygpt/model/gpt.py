"""Decoder-only GPT: token embedding -> N pre-norm blocks -> RMSNorm -> tied unembedding."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tinygpt.model.layers import KV, Block, RMSNorm, Rotary


@dataclass
class GPTConfig:
    """Model hyper-parameters."""

    vocab_size: int = 8192
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    mlp_hidden: int | None = None  # default: 8/3 * d_model rounded up to a multiple of 64
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    attn_impl: str = "sdpa"  # "sdpa" | "manual"

    @property
    def hidden(self) -> int:
        if self.mlp_hidden is not None:
            return self.mlp_hidden
        return 64 * math.ceil(8 * self.d_model / 3 / 64)


class GPT(nn.Module):
    """GPT language model. `forward` returns logits and, given targets, the mean CE loss."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        rotary = Rotary(cfg.d_model // cfg.n_head, cfg.block_size, cfg.rope_theta)
        self.blocks = nn.ModuleList(
            Block(cfg.d_model, cfg.n_head, cfg.hidden, rotary, cfg.attn_impl, cfg.norm_eps)
            for _ in range(cfg.n_layer)
        )
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        """N(0, 0.02) everywhere; residual output projections get std / sqrt(2 * n_layer),
        so each of the 2N residual additions contributes O(1/sqrt(2N)) and the residual
        stream's variance stays ~constant with depth at init (GPT-2)."""
        std = self.cfg.init_std
        if isinstance(m, nn.Linear):
            if getattr(m, "is_residual_out", False):
                std = std / math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(m.weight, 0.0, std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, std)

    def set_attn_impl(self, impl: str) -> None:
        for b in self.blocks:
            b.attn.impl = impl

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past: list[KV] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[KV]]:
        """Args:
            idx: (B, T) token ids.
            targets: optional (B, T) next-token ids for the loss.
            past: optional per-layer KV cache from earlier calls.

        Returns:
            logits (B, T, V), loss (scalar or None), updated KV cache.
        """
        x = self.tok_emb(idx)
        new_past: list[KV] = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, None if past is None else past[i])
            new_past.append(kv)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, new_past

    # ------------------------------------------------------------------ interpretability
    def residual(self, idx: torch.Tensor, layer: int) -> torch.Tensor:
        """Residual stream after `layer` blocks (0 = token embeddings), shape (B, T, D).
        Only the first `layer` blocks are run."""
        x = self.tok_emb(idx)
        for block in self.blocks[:layer]:
            x, _ = block(x)
        return x

    def forward_patched(
        self,
        idx: torch.Tensor,
        layer: int,
        patch: Callable[[torch.Tensor], torch.Tensor],
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with the residual after `layer` blocks replaced by `patch(resid)`.
        Used for SAE splicing, ablations and steering."""
        x = patch(self.residual(idx, layer))
        for block in self.blocks[layer:]:
            x, _ = block(x)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # ------------------------------------------------------------------ accounting
    def num_params(self, non_embedding: bool = True) -> int:
        """Parameter count. Tied weights are counted once. `non_embedding` drops the token
        embedding (= unembedding), following Kaplan et al. (2020)."""
        n = sum(p.numel() for p in self.parameters())  # parameters() dedups tied weights
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def flops_per_token(self, seq_len: int | None = None, include_lm_head: bool = True) -> float:
        """Training FLOPs per token (forward + backward ~ 3x forward).

        6 * N_nonembed for the matmuls, plus 12 * L * D * T for attention scores and
        weighted sums (PaLM appendix B), plus 6 * D * V for the logits when
        `include_lm_head` is set. Used for MFU. The scaling study uses C = 6 N D.
        """
        T = seq_len or self.cfg.block_size
        c = self.cfg
        f = 6 * self.num_params(True) + 12 * c.n_layer * c.d_model * T
        if include_lm_head:
            f += 6 * c.d_model * c.vocab_size
        return float(f)


def load_gpt(path: str | Path, device: str | torch.device = "cpu") -> tuple[GPT, dict[str, Any]]:
    """Load a GPT from a full checkpoint or a weights snapshot; returns (model, ckpt)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
