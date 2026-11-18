"""Global seeding for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU, CUDA, MPS) RNGs.

    Args:
        seed: The seed value.
        deterministic: If True, also request deterministic CUDA kernels. This is slower,
            and a few ops then raise errors, so it stays off for training and on for tests
            that need bit-exactness.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS default generators
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
