"""Full training-state checkpoints: model, optimiser, step, data position, all RNG states."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def rng_state() -> dict[str, Any]:
    """Capture every RNG that training touches."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def save_checkpoint(path: str | Path, **state: Any) -> None:
    """Atomically write a checkpoint (write to a temp file, then rename), so a crash
    mid-write never corrupts the last good checkpoint."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({**state, "rng": rng_state()}, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, restore_rng: bool = True) -> dict[str, Any]:
    """Load a checkpoint dict; optionally restore the global RNG states."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if restore_rng:
        set_rng_state(ckpt["rng"])
    return ckpt
