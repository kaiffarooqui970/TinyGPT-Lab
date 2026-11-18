"""End-to-end integration test on synthetic data (CPU): train a tiny GPT, collect
activations, train TopK and L1 SAEs, and evaluate the CE splice."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tinygpt.model.gpt import GPTConfig
from tinygpt.sae.collect import ActivationStore, CollectConfig, collect
from tinygpt.sae.train_sae import SAETrainConfig, load_sae, train_sae
from tinygpt.train.trainer import TrainConfig, Trainer


def test_tiny_pipeline(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    data.mkdir()
    base = np.tile(np.arange(40), 3000)
    arr = np.where(rng.random(base.size) < 0.1, rng.integers(0, 40, base.size), base)
    arr = arr.astype(np.uint16)
    arr[:100_000].tofile(data / "train_000.bin")
    arr[100_000:].tofile(data / "val_000.bin")

    cfg = TrainConfig(
        name="pipe",
        model=GPTConfig(vocab_size=48, block_size=32, n_layer=2, n_head=2, d_model=32),
        data_dir=str(data),
        tokenizer_path="none.json",
        batch_size=16,
        max_steps=80,
        lr=3e-3,
        warmup_steps=5,
        dtype="fp32",
        device="cpu",
        log_interval=20,
        eval_interval=40,
        eval_batches=2,
        final_eval_batches=4,
        eval_batch_size=8,
        sample_interval=10**6,
        ckpt_interval=10**6,
        runs_root=str(tmp_path / "runs"),
    )
    t = Trainer(cfg)
    t.train()
    t.close()
    ckpt = t.run_dir / "ckpt_final.pt"

    acts_dir = tmp_path / "acts"
    meta = collect(
        CollectConfig(
            checkpoint=str(ckpt),
            layer=1,
            data_dir=str(data),
            n_tokens=20_000,
            batch_size=16,
            out_dir=str(acts_dir),
            shard_tokens=8_000,
            device="cpu",
        )
    )
    store = ActivationStore(acts_dir)
    a, tok = store.load_all()
    assert a.dtype == torch.float16 and a.shape == (meta["n_tokens"], 32)
    assert tok.shape == (meta["n_tokens"],) and meta["n_tokens"] % 32 == 0
    assert len(meta["shard_sizes"]) >= 2 and meta["norm_mean"] > 0

    results = {}
    for kind in ("topk", "l1"):
        sae, m = train_sae(
            SAETrainConfig(
                name=kind,
                acts_dir=str(acts_dir),
                kind=kind,
                expansion=4,
                k=8,
                l1_coef=3e-3,
                train_tokens=60_000,
                heldout_tokens=2_048,
                batch_size=512,
                lr=2e-3,
                dead_tokens=5_000,
                log_interval=20,
                ce_eval_batches=2,
                ce_eval_batch_size=8,
                data_dir=str(data),
                device="cpu",
                runs_root=str(tmp_path / "sae"),
            )
        )
        results[kind] = m
        assert 0.0 < m["fve"] <= 1.0
        assert m["ce_zero"] > m["ce_clean"]
    assert results["topk"]["l0"] <= 8 + 1e-6
    assert results["topk"]["ce_recovered"] > 0.5
    run = next((tmp_path / "sae").glob("*_topk"))
    sae2, _ = load_sae(run / "sae.pt")
    assert json.loads((run / "final.json").read_text())["kind"] == "topk"
    assert sae2.n_latents == 128
