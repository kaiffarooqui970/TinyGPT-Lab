"""Phase 3 acceptance tests for the GPT model."""

from __future__ import annotations

import pytest
import torch

from tinygpt.model.generate import filter_logits, generate
from tinygpt.model.gpt import GPT, GPTConfig
from tinygpt.utils.seed import seed_everything


def tiny(**kw: object) -> GPT:
    seed_everything(0)
    cfg = GPTConfig(vocab_size=97, block_size=32, n_layer=2, n_head=4, d_model=32, **kw)
    return GPT(cfg)


def test_forward_shapes_and_loss() -> None:
    m = tiny()
    x = torch.randint(0, 97, (3, 16))
    y = torch.randint(0, 97, (3, 16))  # independent targets: tied embeddings favour y == x
    logits, loss, past = m(x, y)
    assert logits.shape == (3, 16, 97)
    assert loss is not None and abs(loss.item() - torch.log(torch.tensor(97.0)).item()) < 0.3
    assert len(past) == 2 and past[0][0].shape == (3, 4, 16, 8)


@pytest.mark.parametrize("impl", ["sdpa", "manual"])
def test_causality(impl: str) -> None:
    """Changing token t+1 must not change logits at positions <= t."""
    m = tiny(attn_impl=impl).eval()
    x = torch.randint(0, 97, (2, 20))
    for t in [0, 5, 18]:
        x2 = x.clone()
        x2[:, t + 1 :] = (x2[:, t + 1 :] + 1) % 97
        a, _, _ = m(x)
        b, _, _ = m(x2)
        assert torch.equal(a[:, : t + 1], b[:, : t + 1])
        assert not torch.allclose(a[:, t + 1], b[:, t + 1])


def test_manual_matches_sdpa() -> None:
    m = tiny().eval()
    x = torch.randint(0, 97, (2, 32))
    with torch.no_grad():
        m.set_attn_impl("sdpa")
        a, _, _ = m(x)
        m.set_attn_impl("manual")
        b, _, _ = m(x)
    assert (a - b).abs().max().item() < 1e-5


@pytest.mark.parametrize("impl", ["sdpa", "manual"])
def test_cached_forward_matches_full(impl: str) -> None:
    m = tiny(attn_impl=impl).eval()
    x = torch.randint(0, 97, (2, 12))
    with torch.no_grad():
        full, _, _ = m(x)
        _, _, past = m(x[:, :7])
        part, _, _ = m(x[:, 7:], past=past)  # multi-token chunk with offset mask
    assert (full[:, 7:] - part).abs().max().item() < 1e-5


@pytest.mark.parametrize("impl", ["sdpa", "manual"])
def test_kv_cache_greedy_matches_no_cache(impl: str) -> None:
    m = tiny(attn_impl=impl)
    prompt = torch.randint(0, 97, (2, 5))
    a = generate(m, prompt, 20, temperature=0.0, use_cache=True)
    b = generate(m, prompt, 20, temperature=0.0, use_cache=False)
    assert torch.equal(a, b)


def test_sampling_reproducible_and_filters() -> None:
    m = tiny()
    p = torch.randint(0, 97, (1, 3))
    a = generate(m, p, 10, temperature=0.8, top_k=10, generator=torch.Generator().manual_seed(1))
    b = generate(m, p, 10, temperature=0.8, top_k=10, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    assert torch.isinf(filter_logits(logits, 2, None)[0, 2:]).all()
    # probs ~ [.64, .24, .09, .03]: top_p=0.8 keeps the first two
    kept = torch.isfinite(filter_logits(logits, None, 0.8))[0].tolist()
    assert kept == [True, True, False, False]


def test_param_count_and_tying() -> None:
    m = tiny()
    assert m.lm_head.weight is m.tok_emb.weight
    d, h, L, V = 32, m.cfg.hidden, 2, 97
    per_layer = 4 * d * d + 3 * d * h + 2 * d
    assert m.num_params(non_embedding=True) == L * per_layer + d
    assert m.num_params(non_embedding=False) == L * per_layer + d + V * d
    assert all(
        not hasattr(mod, "bias") or mod.bias is None
        for mod in m.modules()
        if isinstance(mod, torch.nn.Linear)
    )


def test_residual_init_scaled() -> None:
    m = tiny()
    std_res = m.blocks[0].mlp.w2.weight.std().item()
    std_in = m.blocks[0].mlp.w1.weight.std().item()
    assert std_res == pytest.approx(0.02 / 2.0, rel=0.15)  # 2 layers -> /sqrt(4)
    assert std_in == pytest.approx(0.02, rel=0.15)


def test_flops_per_token() -> None:
    m = tiny()
    n = m.num_params(True)
    assert m.flops_per_token(32, include_lm_head=False) == 6 * n + 12 * 2 * 32 * 32


def test_overfit_single_batch() -> None:
    m = tiny()
    x = torch.randint(0, 97, (4, 32))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(300):
        _, loss, _ = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
