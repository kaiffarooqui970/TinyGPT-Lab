"""Transformer building blocks: RMSNorm, RoPE, causal attention (manual or SDPA), SwiGLU.

Shapes: B = batch, T = query length, S = key length (T + cached length),
H = heads, Dh = head dim, D = model dim.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

KV = tuple[torch.Tensor, torch.Tensor]


class RMSNorm(nn.Module):
    """y = x / rms(x) * g. Unlike LayerNorm it neither centres nor uses a bias; it is cheaper
    and works as well in pre-norm transformers (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise in fp32 for stability under bf16 autocast, then cast back.
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.type_as(x) * self.weight


class Rotary(nn.Module):
    """Rotary position embedding (Su et al., 2021), using the 'rotate-half' layout.

    Each pair of channels (i, i + Dh/2) is rotated by angle pos * theta^(-2i/Dh). The q·k
    dot product then depends only on the relative offset between positions.
    """

    def __init__(self, head_dim: int, max_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_len, Dh/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Rotate x of shape (B, H, T, Dh) for positions offset..offset+T-1."""
        T = x.shape[-2]
        if offset + T > self.cos.shape[0]:
            raise ValueError(f"position {offset + T} exceeds RoPE table {self.cos.shape[0]}")
        cos = self.cos[offset : offset + T].to(x.dtype)
        sin = self.sin[offset : offset + T].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal softmax(q k^T / sqrt(Dh)) v, with queries aligned to the *last* T keys.

    Query i (0-based, of T) sits at absolute position S - T + i and may attend to keys
    0..S - T + i. That covers training (S == T) and cached decoding (S > T).
    """
    T, S = q.shape[-2], k.shape[-2]
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
    mask = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(diagonal=S - T)
    att = att.masked_fill(~mask, float("-inf"))
    att = F.softmax(att.float(), dim=-1).type_as(q)
    return att @ v


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Same maths via the fused kernel. `is_causal` assumes S == T, so the cached case
    builds an explicit offset mask (or needs none when T == 1)."""
    T, S = q.shape[-2], k.shape[-2]
    if T == S:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if T == 1:
        return F.scaled_dot_product_attention(q, k, v)
    mask = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(diagonal=S - T)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and an optional KV cache. No biases."""

    def __init__(self, d_model: int, n_head: int, rotary: Rotary, impl: str = "sdpa") -> None:
        super().__init__()
        if d_model % n_head:
            raise ValueError("d_model must be divisible by n_head")
        if impl not in ("sdpa", "manual"):
            raise ValueError(f"unknown attention impl {impl!r}")
        self.n_head, self.head_dim, self.impl = n_head, d_model // n_head, impl
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj.is_residual_out = True  # type: ignore[attr-defined]  # scaled init
        self.rotary = rotary

    def forward(self, x: torch.Tensor, past: KV | None = None) -> tuple[torch.Tensor, KV]:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        offset = 0 if past is None else past[0].shape[-2]
        q, k = self.rotary(q, offset), self.rotary(k, offset)
        if past is not None:
            k = torch.cat([past[0], k], dim=-2)
            v = torch.cat([past[1], v], dim=-2)
        attn = sdpa_attention if self.impl == "sdpa" else manual_attention
        y = attn(q, k, v).transpose(1, 2).reshape(B, T, D)
        return self.proj(y), (k, v)


class SwiGLU(nn.Module):
    """MLP(x) = W2 (silu(W1 x) * W3 x) (Shazeer, 2020). Hidden size ~ 8/3 * D, which keeps
    the parameter count equal to a 4*D GELU MLP."""

    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)  # gate
        self.w3 = nn.Linear(d_model, hidden, bias=False)  # up
        self.w2 = nn.Linear(hidden, d_model, bias=False)  # down
        self.w2.is_residual_out = True  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """Pre-norm residual block: x += attn(norm(x)); x += mlp(norm(x))."""

    def __init__(
        self, d_model: int, n_head: int, hidden: int, rotary: Rotary, impl: str, eps: float
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps)
        self.attn = CausalSelfAttention(d_model, n_head, rotary, impl)
        self.norm2 = RMSNorm(d_model, eps)
        self.mlp = SwiGLU(d_model, hidden)

    def forward(self, x: torch.Tensor, past: KV | None = None) -> tuple[torch.Tensor, KV]:
        a, kv = self.attn(self.norm1(x), past)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, kv
