"""Phase 8 tests: feature stats, dashboard output, ablation and steering mechanics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tinygpt.interp.dashboard import build_feature_records, write_dashboard
from tinygpt.interp.features import compute_feature_stats, logit_effects
from tinygpt.interp.interventions import (
    ablate_fn,
    bootstrap_diff_ci,
    decoder_dir,
    steering,
)
from tinygpt.model.gpt import GPT, GPTConfig
from tinygpt.sae.sae import SAE, SAEConfig
from tinygpt.tokenizer.bpe import EOT, BPETokenizer


def setup() -> tuple[GPT, SAE, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=300, block_size=16, n_layer=2, n_head=2, d_model=16)).eval()
    x = torch.randint(0, 300, (8, 16))
    acts = model.residual(x, 1).detach().reshape(-1, 16)
    sae = SAE(SAEConfig(d_in=16, expansion=4, kind="topk", k=4))
    sae.set_input_stats(acts)
    return model, sae, acts, x.reshape(-1)


def test_feature_stats_topk_matches_bruteforce() -> None:
    _, sae, acts, _ = setup()
    st = compute_feature_stats(sae, acts, n_top=5, batch_size=37)  # odd batch on purpose
    z = sae.encode(acts).detach()
    for i in range(sae.n_latents):
        best = torch.topk(z[:, i], 5).values.numpy()
        assert np.allclose(st.top_vals[:, i], best, atol=1e-5)
    assert np.allclose(st.freq, (z > 0).float().mean(0).numpy())
    assert st.hist.sum() == (z > 0).sum().item()


def test_dashboard_writes_html(tmp_path: Path) -> None:
    model, sae, acts, toks = setup()
    tok = BPETokenizer.train(["hello world " * 50], vocab_size=299, special_tokens=[EOT])
    st = compute_feature_stats(sae, acts, n_top=4)
    feats = [0, 1, 2]
    recs = build_feature_records(
        feats,
        st,
        sae,
        acts,
        toks.clamp_max(tok.vocab_size - 1),
        16,
        tok,
        logit_effects(model, sae, feats, top=3),
        n_examples=3,
    )
    out = tmp_path / "d.html"
    write_dashboard(out, recs, "test", {"k": 4})
    text = out.read_text()
    assert text.startswith("<!doctype html>") and '"id": 0' in text


def test_ablating_all_active_features_equals_removing_reconstruction() -> None:
    _, sae, acts, _ = setup()
    r = acts[:10]
    out = ablate_fn(sae, range(sae.n_latents))(r)
    expected = r - (sae.decode(sae.encode(r)) - sae.b_dec).detach() / sae.scale
    assert torch.allclose(out, expected, atol=1e-5)


def test_steering_hook_adds_vector_and_is_removed() -> None:
    model, sae, _, toks = setup()
    x = toks[:16][None]
    vec = decoder_dir(sae, 3) * 5.0
    with torch.no_grad():
        base = model.residual(x, 2)
        with steering(model, 1, vec):
            logits_s, _, _ = model(x)
        manual, _ = model.forward_patched(x, 1, lambda r: r + vec)
        logits_after, _, _ = model(x)
        clean, _, _ = model(x)
    assert torch.allclose(logits_s, manual, atol=1e-5)
    assert torch.allclose(logits_after, clean) and base.shape == (1, 16, 16)


def test_bootstrap_diff_ci() -> None:
    rng = np.random.default_rng(0)
    r = bootstrap_diff_ci(rng.normal(1, 1, 400), rng.normal(0, 1, 400))
    assert r["ci_lo"] < 1.0 < r["ci_hi"] and 0.7 < r["cohens_d"] < 1.3


def test_max_correlation_recovers_permuted_latents() -> None:
    from tinygpt.interp.universality import max_correlation, summarise

    g = torch.Generator().manual_seed(0)
    zA = torch.relu(torch.randn(2000, 30, generator=g))
    perm = torch.randperm(30, generator=g)
    zB = torch.cat(
        [zA[:, perm] * 3.0, torch.relu(torch.randn(2000, 10, generator=g)), torch.zeros(2000, 1)],
        dim=1,
    )  # 30 copies, 10 novel, 1 dead
    res = max_correlation(zA.half(), zB.half(), chunk=7)
    mc = res["max_corr"]
    assert np.allclose(mc[:30], 1.0, atol=1e-2)
    assert np.all(res["argmax"][:30] == perm.numpy())
    assert np.all(mc[30:40] < 0.2) and np.isnan(mc[40])
    s = summarise(mc)
    assert s["n_alive"] == 40 and abs(s["frac_ge_0.9"] - 0.75) < 1e-6
