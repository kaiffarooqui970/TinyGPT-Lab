"""Throughput and model-FLOPs-utilisation (MFU) helpers."""

from __future__ import annotations

import torch

# Dense peak throughput (FLOP/s) at the dtype used for training. They are nominal figures,
# so MFU is a rough efficiency indicator, not an exact measurement.
PEAK_FLOPS: dict[str, float] = {
    "A100": 312e12,  # bf16
    "H100": 989e12,  # bf16, SXM
    "4090": 165e12,  # bf16 tensor core, fp32 accumulate
    "A10": 125e12,
    "L4": 121e12,
    "mps": 4.3e12,  # Apple M4 10-core GPU, fp32 (approximate)
    "cpu": 1e12,  # placeholder: MFU is not meaningful on CPU
}


def device_peak_flops(device: torch.device, override: float | None = None) -> float:
    """Best-guess peak FLOP/s for `device` (`override` wins if given)."""
    if override is not None:
        return override
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        for key, val in PEAK_FLOPS.items():
            if key in name:
                return val
        return 100e12
    return PEAK_FLOPS.get(device.type, 1e12)


def mfu(flops_per_token: float, tokens_per_sec: float, peak: float) -> float:
    """Fraction of the hardware's peak FLOP/s spent on model FLOPs."""
    return flops_per_token * tokens_per_sec / peak


def pick_device(name: str = "auto") -> torch.device:
    """'auto' picks cuda > mps > cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
