"""Random-window loader over uint16 memmap shards.

Each sample is a contiguous window of `block_size + 1` tokens inside one shard:
x = window[:-1], y = window[1:]. A shard is picked with probability proportional to
its number of valid start positions, so every start position across the dataset is
equally likely. All randomness comes from one `torch.Generator`, so the data order is
fixed by `seed`, is identical for every model size, and can be checkpointed exactly.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


class TokenLoader:
    """Samples (x, y) batches of shape (batch_size, block_size) from token shards."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        block_size: int,
        batch_size: int,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.block_size, self.batch_size, self.device = block_size, batch_size, device
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        starts = np.array([max(len(s) - block_size, 0) for s in self.shards], dtype=np.int64)
        if starts.sum() == 0:
            raise ValueError("no shard is longer than block_size")
        self._cum = np.cumsum(starts)  # global start index -> (shard, offset)
        self._n_starts = int(self._cum[-1])
        self.gen = torch.Generator().manual_seed(seed)
        self.step = 0

    @property
    def total_tokens(self) -> int:
        return int(sum(len(s) for s in self.shards))

    def _windows(self, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        shard_ids = np.searchsorted(self._cum, idx, side="right")
        prev = np.concatenate([[0], self._cum[:-1]])
        rows = []
        for g, s in zip(idx, shard_ids, strict=True):
            off = int(g - prev[s])
            rows.append(np.asarray(self.shards[s][off : off + self.block_size + 1], np.int64))
        buf = torch.from_numpy(np.stack(rows))
        x, y = buf[:, :-1], buf[:, 1:]
        if self.device != "cpu":
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        return x, y

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Next training batch; advances the generator and the step counter."""
        idx = torch.randint(self._n_starts, (self.batch_size,), generator=self.gen).numpy()
        self.step += 1
        return self._windows(idx)

    def fixed_batches(
        self, n_batches: int, seed: int = 1234
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """A fixed, repeatable set of batches (for evaluation), independent of training."""
        g = torch.Generator().manual_seed(seed)
        for _ in range(n_batches):
            idx = torch.randint(self._n_starts, (self.batch_size,), generator=g).numpy()
            yield self._windows(idx)

    def state_dict(self) -> dict[str, Any]:
        return {"gen_state": self.gen.get_state(), "step": self.step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.gen.set_state(state["gen_state"])
        self.step = int(state["step"])
