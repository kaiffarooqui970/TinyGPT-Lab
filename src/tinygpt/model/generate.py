"""Autoregressive sampling with KV cache, temperature, top-k and top-p (nucleus)."""

from __future__ import annotations

import torch

from tinygpt.model.gpt import GPT


def filter_logits(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    """Mask logits outside the top-k and/or the smallest nucleus with mass >= top_p."""
    if top_k is not None and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(-1)
        # Drop a token if the mass *before* it already reaches top_p (always keeps the top one).
        drop = probs.cumsum(-1) - probs >= top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Extend `idx` (B, T) by up to `max_new_tokens`. temperature == 0 means greedy decoding.

    With the cache, the prompt is processed once and then one token per step. The context
    is limited to `block_size` (RoPE table length). Without the cache, the full
    (cropped) sequence is recomputed every step.
    """
    model.eval()
    block = model.cfg.block_size
    past = None
    out = idx
    for _ in range(max_new_tokens):
        if use_cache:
            if past is not None and past[0][0].shape[-2] >= block:
                past = None  # context full: restart cache from the last block-1 tokens
            inp = out[:, -(block - 1) :] if past is None else out[:, -1:]
            logits, _, past = model(inp, past=past)
        else:
            logits, _, _ = model(out[:, -block:])
        logits = logits[:, -1, :].float()
        if temperature == 0.0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = filter_logits(logits / temperature, top_k, top_p)
            probs = logits.softmax(-1)
            if generator is not None and generator.device != probs.device:
                # A seeded CPU generator gives the same samples on any device.
                nxt = torch.multinomial(probs.cpu(), 1, generator=generator).to(probs.device)
            else:
                nxt = torch.multinomial(probs, 1, generator=generator)
        out = torch.cat([out, nxt], dim=1)
        if eos_id is not None and bool((nxt == eos_id).all()):
            break
    return out
