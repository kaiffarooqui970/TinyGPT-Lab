"""Measure training throughput (tokens/s) for several model shapes on the current device.

Usage:
    uv run python scripts/bench_throughput.py
"""

from __future__ import annotations

import contextlib
import json
import time

import torch

from tinygpt.model.gpt import GPT, GPTConfig
from tinygpt.train.metrics import pick_device
from tinygpt.train.optim import make_optimizer

SHAPES = [  # (n_layer, d_model, n_head)
    (2, 128, 4),
    (4, 192, 6),
    (6, 256, 8),
    (6, 384, 6),
    (8, 512, 8),
]


def bench(
    n_layer: int, d: int, h: int, dtype: str, bs: int = 32, T: int = 256, steps: int = 8
) -> dict:
    dev = pick_device()
    m = GPT(GPTConfig(n_layer=n_layer, d_model=d, n_head=h, block_size=T)).to(dev)
    opt = make_optimizer(m, 1e-3, 0.1, (0.9, 0.95))
    ctx = (
        torch.autocast(dev.type, dtype=torch.bfloat16)
        if dtype == "bf16"
        else contextlib.nullcontext()
    )
    x = torch.randint(0, 8192, (bs, T), device=dev)

    def step() -> None:
        with ctx:
            _, loss, _ = m(x, x)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(2):
        step()
    torch.mps.synchronize() if dev.type == "mps" else None
    t = time.perf_counter()
    for _ in range(steps):
        step()
    torch.mps.synchronize() if dev.type == "mps" else None
    tps = bs * T * steps / (time.perf_counter() - t)
    return {
        "n_layer": n_layer,
        "d_model": d,
        "N_nonembed": m.num_params(True),
        "dtype": dtype,
        "tokens_per_sec": round(tps),
    }


if __name__ == "__main__":
    for shape in SHAPES:
        for dtype in ("fp32", "bf16"):
            try:
                print(json.dumps(bench(*shape, dtype)), flush=True)
            except Exception as e:  # report, don't hide, unsupported dtypes
                print(json.dumps({"shape": shape, "dtype": dtype, "error": str(e)[:200]}))
