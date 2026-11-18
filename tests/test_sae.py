"""Phase 7 tests for the sparse autoencoders."""

from __future__ import annotations

import pytest
import torch

from tinygpt.sae.sae import SAE, SAEConfig


def make(kind: str = "topk", **kw: object) -> SAE:
    torch.manual_seed(0)
    cfg = SAEConfig(d_in=16, expansion=8, kind=kind, k=4, **kw)  # type: ignore[arg-type]
    sae = SAE(cfg)
    sae.set_input_stats(torch.randn(512, 16) * 3 + 1)
    return sae


def test_decoder_columns_unit_norm_at_init_and_after_step() -> None:
    sae = make()
    assert torch.allclose(sae.W_dec.norm(dim=0), torch.ones(sae.n_latents), atol=1e-5)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
    x = torch.randn(64, 16) * 3 + 1
    for _ in range(5):
        out = sae(x)
        out.loss.backward()
        sae.project_decoder_grad()
        opt.step()
        opt.zero_grad()
        sae.normalize_decoder()
    assert torch.allclose(sae.W_dec.norm(dim=0), torch.ones(sae.n_latents), atol=1e-5)


def test_decoder_grad_projection_removes_parallel_component() -> None:
    sae = make()
    sae(torch.randn(32, 16)).loss.backward()
    sae.project_decoder_grad()
    dots = (sae.W_dec.grad * sae.W_dec).sum(0)
    assert dots.abs().max().item() < 1e-6


def test_topk_exactly_k_active() -> None:
    sae = make()
    with torch.no_grad():
        sae.b_enc.fill_(10.0)  # force all pre-activations positive
        z = sae.encode(torch.randn(50, 16))
    assert torch.all((z != 0).sum(-1) == 4)


def test_topk_at_most_k_active_in_general() -> None:
    sae = make()
    z = sae.encode(torch.randn(50, 16))
    assert torch.all((z != 0).sum(-1) <= 4)
    assert torch.all(z >= 0)


def test_l1_variant_and_input_normalisation_roundtrip() -> None:
    sae = make("l1", l1_coef=1e-2)
    x = torch.randn(8, 16) * 3 + 1
    out = sae(x)
    assert out.loss.item() > 0 and out.l1 is not None
    # The normalised -> denormalised reconstruction is in the original units.
    assert sae.reconstruct(x).shape == x.shape
    # The scale makes E||x * s||^2 = d_in on the stats data (Gao et al. 2024 convention).
    stats = torch.randn(4096, 16) * 3 + 1
    sae.set_input_stats(stats)
    assert (stats * sae.scale).pow(2).sum(-1).mean().item() == pytest.approx(16, rel=1e-4)


def test_auxk_loss_only_with_dead_latents() -> None:
    sae = make()
    x = torch.randn(64, 16)
    out = sae(x, dead_mask=torch.zeros(sae.n_latents, dtype=torch.bool))
    assert out.aux_loss.item() == 0.0
    dead = torch.zeros(sae.n_latents, dtype=torch.bool)
    dead[:40] = True
    out = sae(x, dead_mask=dead)
    assert out.aux_loss.item() > 0.0


def test_perfect_fit_possible() -> None:
    """With k = d_in and enough steps, a TopK SAE reconstructs low-rank data well."""
    torch.manual_seed(0)
    basis = torch.randn(4, 16)
    codes = torch.relu(torch.randn(2048, 4))
    x = codes @ basis
    sae = SAE(SAEConfig(d_in=16, expansion=4, kind="topk", k=4))
    sae.set_input_stats(x)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    for _ in range(600):
        out = sae(x)
        out.loss.backward()
        sae.project_decoder_grad()
        opt.step()
        opt.zero_grad()
        sae.normalize_decoder()
    assert out.fvu.item() < 0.05


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_metrics_and_feature_stats_run_on_mps() -> None:
    from tinygpt.interp.features import compute_feature_stats
    from tinygpt.sae.eval_sae import activation_metrics

    sae = make()
    acts = torch.randn(300, 16) * 3 + 1
    m = activation_metrics(sae, acts, batch_size=128, device="mps")
    m_cpu = activation_metrics(make(), acts, batch_size=128, device="cpu")
    assert abs(m["fve"] - m_cpu["fve"]) < 1e-3
    st = compute_feature_stats(make(), acts, n_top=3, device="mps")
    assert st.hist.sum() > 0
