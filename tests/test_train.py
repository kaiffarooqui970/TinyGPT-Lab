"""Phase 4 tests: optimiser groups, LR schedule, bit-identical resume, a short training run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tinygpt.model.gpt import GPT, GPTConfig
from tinygpt.train.optim import lr_at, param_groups
from tinygpt.train.trainer import TrainConfig, Trainer
from tinygpt.utils.config import load_config


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    d = tmp_path / "data"
    d.mkdir()
    # A learnable sequence: repeating pattern with noise.
    base = np.tile(np.arange(50), 2000)
    noise = rng.integers(0, 50, base.size)
    arr = np.where(rng.random(base.size) < 0.1, noise, base).astype(np.uint16)
    arr[:90_000].tofile(d / "train_000.bin")
    arr[90_000:].tofile(d / "val_000.bin")
    return d


def tiny_cfg(data_dir: Path, tmp_path: Path, **kw: object) -> TrainConfig:
    base = dict(
        name="t",
        model=GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, d_model=32),
        data_dir=str(data_dir),
        tokenizer_path="none.json",
        batch_size=8,
        grad_accum=2,
        max_steps=8,
        lr=3e-3,
        warmup_steps=2,
        dtype="fp32",
        device="cpu",
        log_interval=1,
        eval_interval=4,
        eval_batches=2,
        final_eval_batches=2,
        eval_batch_size=4,
        sample_interval=10_000,
        ckpt_interval=10_000,
        runs_root=str(tmp_path / "runs"),
    )
    base.update(kw)
    return TrainConfig(**base)  # type: ignore[arg-type]


def test_param_groups_exclude_norms_and_embeddings() -> None:
    m = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=16))
    decay, no_decay = param_groups(m, 0.1)
    assert all(p.ndim == 2 for p in decay["params"])
    assert any(p is m.tok_emb.weight for p in no_decay["params"])
    assert all(p.ndim == 1 or p is m.tok_emb.weight for p in no_decay["params"])
    n = sum(p.numel() for g in (decay, no_decay) for p in g["params"])
    assert n == m.num_params(non_embedding=False)  # every parameter is in exactly one group


def test_lr_schedule() -> None:
    assert lr_at(0, 1.0, 10, 100) == pytest.approx(0.1)
    assert lr_at(9, 1.0, 10, 100) == pytest.approx(1.0)
    assert lr_at(10, 1.0, 10, 100) == pytest.approx(1.0)
    assert lr_at(100, 1.0, 10, 100) == pytest.approx(0.1)
    mid = lr_at(55, 1.0, 10, 100)
    assert 0.5 < mid < 0.6  # halfway through cosine: (1 + 0.1) / 2 = 0.55


def test_resume_is_bit_identical(data_dir: Path, tmp_path: Path) -> None:
    cfg = tiny_cfg(data_dir, tmp_path)
    a = Trainer(cfg)
    a.train()
    ref = {k: v.clone() for k, v in a.model.state_dict().items()}

    b = Trainer(cfg)
    b.train(stop_at=3)
    ckpt = b.run_dir / "resume.pt"
    b.save(ckpt)
    torch.manual_seed(12345)  # scramble global RNG to prove it gets restored
    c = Trainer(cfg, resume_from=ckpt)
    assert c.step == 3
    c.train()
    for k, v in c.model.state_dict().items():
        assert torch.equal(v, ref[k]), k


def test_training_reduces_loss_and_writes_outputs(data_dir: Path, tmp_path: Path) -> None:
    cfg = tiny_cfg(data_dir, tmp_path, max_steps=60, eval_interval=30)
    t = Trainer(cfg)
    result = t.train()
    rows = [json.loads(line) for line in (t.run_dir / "metrics.jsonl").read_text().splitlines()]
    train_losses = [r["train_loss"] for r in rows if "train_loss" in r]
    assert train_losses[-1] < train_losses[0] - 1.0
    assert {"lr", "grad_norm", "tokens_per_sec", "mfu"} <= rows[0].keys()
    assert (t.run_dir / "final.json").exists() and (t.run_dir / "ckpt_final.pt").exists()
    assert len(result["val_loss_per_batch"]) == 2


def test_train_length_from_tokens_per_param(data_dir: Path, tmp_path: Path) -> None:
    cfg = tiny_cfg(data_dir, tmp_path, max_steps=None, tokens_per_param=2.0)
    t = Trainer(cfg)
    assert t.total_steps == -(-int(2.0 * t.n_params) // t.tokens_per_step)


def test_repo_train_configs_parse() -> None:
    root = Path(__file__).resolve().parents[1]
    for p in sorted((root / "configs" / "train").glob("*.yaml")):
        load_config(p, TrainConfig)
