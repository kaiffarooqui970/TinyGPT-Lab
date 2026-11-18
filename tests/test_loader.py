"""Phase 2 acceptance tests for the memmap token loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tinygpt.data.loader import TokenLoader


@pytest.fixture
def shards(tmp_path: Path) -> list[Path]:
    paths = []
    start = 0
    for i, n in enumerate([1000, 37, 2500]):  # includes a shard shorter than a window
        p = tmp_path / f"train_{i:03d}.bin"
        (np.arange(start, start + n) % 60000).astype(np.uint16).tofile(p)
        start += n
        paths.append(p)
    return paths


def test_shapes_dtype_and_shift(shards: list[Path]) -> None:
    dl = TokenLoader(shards, block_size=64, batch_size=8, seed=0)
    x, y = dl.next_batch()
    assert x.shape == (8, 64) and y.shape == (8, 64)
    assert x.dtype == torch.long and y.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])  # targets are inputs shifted by one
    # Data is arange, so every window must be contiguous (never crosses a shard boundary).
    assert torch.all(y - x == 1)


def test_deterministic_order(shards: list[Path]) -> None:
    a = TokenLoader(shards, block_size=32, batch_size=4, seed=7)
    b = TokenLoader(shards, block_size=32, batch_size=4, seed=7)
    for _ in range(5):
        assert torch.equal(a.next_batch()[0], b.next_batch()[0])
    c = TokenLoader(shards, block_size=32, batch_size=4, seed=8)
    assert not torch.equal(a.next_batch()[0], c.next_batch()[0])


def test_state_dict_resume(shards: list[Path]) -> None:
    a = TokenLoader(shards, block_size=32, batch_size=4, seed=1)
    for _ in range(3):
        a.next_batch()
    state = a.state_dict()
    expected = [a.next_batch()[0] for _ in range(3)]
    b = TokenLoader(shards, block_size=32, batch_size=4, seed=999)
    b.load_state_dict(state)
    got = [b.next_batch()[0] for _ in range(3)]
    assert all(torch.equal(e, g) for e, g in zip(expected, got, strict=True))
    assert b.step == a.step


def test_fixed_eval_batches_repeatable(shards: list[Path]) -> None:
    dl = TokenLoader(shards, block_size=16, batch_size=4, seed=0)
    e1 = [x for x, _ in dl.fixed_batches(3, seed=123)]
    dl.next_batch()  # training draws must not affect the eval set
    e2 = [x for x, _ in dl.fixed_batches(3, seed=123)]
    assert all(torch.equal(a, b) for a, b in zip(e1, e2, strict=True))


def test_total_tokens(shards: list[Path]) -> None:
    dl = TokenLoader(shards, block_size=16, batch_size=2, seed=0)
    assert dl.total_tokens == 1000 + 37 + 2500
