"""Record residual-stream activations from a trained model into fp16 shards.

Output directory:
    acts_000.bin    float16, shape (n, d_model), row-major
    tokens_000.bin  uint16,  shape (n,): the token id at each activation's position
    meta.json       d_model, layer, block_size, shard sizes, norm statistics

Activations come from whole windows of `block_size` tokens, in order, so row r belongs
to window r // block_size at position r % block_size. That lets us recover the context
of any activation for the dashboards.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tinygpt.data.loader import TokenLoader
from tinygpt.data.prepare import resolve, shard_paths
from tinygpt.model.gpt import load_gpt
from tinygpt.train.metrics import pick_device


@dataclass
class CollectConfig:
    checkpoint: str
    layer: int = 3  # residual stream after this many blocks
    data_dir: str = "data/tinystories"
    split: str = "train"
    n_tokens: int = 4_000_000
    batch_size: int = 64
    data_seed: int = 777  # different from the training order
    out_dir: str = "data/acts/s5_l3"
    shard_tokens: int = 1_048_576
    device: str = "auto"
    seed: int = 0


def iter_activations(cfg: CollectConfig) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (acts fp16 (B*T, D), tokens uint16 (B*T,)) chunks until n_tokens is reached."""
    device = pick_device(cfg.device)
    model, _ = load_gpt(resolve(cfg.checkpoint), device)
    T = model.cfg.block_size
    loader = TokenLoader(
        shard_paths(cfg.data_dir, cfg.split), T, cfg.batch_size, seed=cfg.data_seed
    )
    done = 0
    while done < cfg.n_tokens:
        x, _ = loader.next_batch()
        with torch.no_grad():
            r = model.residual(x.to(device), cfg.layer).float()
        take = min(x.numel(), cfg.n_tokens - done)
        take -= take % T  # keep whole windows
        if take <= 0:
            break
        acts = r.reshape(-1, r.shape[-1])[:take].cpu().numpy().astype(np.float16)
        toks = x.reshape(-1)[:take].numpy().astype(np.uint16)
        done += take
        yield acts, toks


def collect(cfg: CollectConfig) -> dict:
    """Run collection and write shards + meta; return the meta dict."""
    out = resolve(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("acts_*.bin")) + list(out.glob("tokens_*.bin")):
        old.unlink()
    shard_sizes: list[int] = []
    fa = ft = None
    in_shard = 0
    # Running statistics, accumulated in float64.
    n = 0
    s_norm = s_norm2 = 0.0
    mean = None
    per_pos_norm = None
    max_norm = 0.0
    d = 0
    T = _block_size(cfg)
    for acts, toks in iter_activations(cfg):
        a32 = acts.astype(np.float32)
        norms = np.linalg.norm(a32, axis=1)
        d = acts.shape[1]
        n += len(norms)
        s_norm += float(norms.sum())
        s_norm2 += float((norms.astype(np.float64) ** 2).sum())
        max_norm = max(max_norm, float(norms.max()))
        mean = a32.sum(0, dtype=np.float64) if mean is None else mean + a32.sum(0, np.float64)
        pp = norms.reshape(-1, T).sum(0)
        per_pos_norm = pp if per_pos_norm is None else per_pos_norm + pp
        i = 0
        while i < len(acts):
            if fa is None:
                idx = len(shard_sizes)
                fa = open(out / f"acts_{idx:03d}.bin", "wb")  # noqa: SIM115
                ft = open(out / f"tokens_{idx:03d}.bin", "wb")  # noqa: SIM115
                shard_sizes.append(0)
            take = min(len(acts) - i, cfg.shard_tokens - in_shard)
            acts[i : i + take].tofile(fa)
            toks[i : i + take].tofile(ft)
            in_shard += take
            shard_sizes[-1] += take
            i += take
            if in_shard == cfg.shard_tokens:
                fa.close()
                ft.close()
                fa = ft = None
                in_shard = 0
    if fa is not None:
        fa.close()
        ft.close()
    assert mean is not None and per_pos_norm is not None
    meta = {
        "checkpoint": cfg.checkpoint,
        "layer": cfg.layer,
        "d_model": d,
        "block_size": T,
        "n_tokens": n,
        "shard_sizes": shard_sizes,
        "dtype": "float16",
        "split": cfg.split,
        "norm_mean": s_norm / n,
        "norm_std": float(np.sqrt(s_norm2 / n - (s_norm / n) ** 2)),
        "norm_max": max_norm,
        "mean_vector_norm": float(np.linalg.norm(mean / n)),
        "norm_by_position": (per_pos_norm / (n / T)).tolist(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _block_size(cfg: CollectConfig) -> int:
    ckpt = torch.load(resolve(cfg.checkpoint), map_location="cpu", weights_only=False)
    return int(ckpt["model_cfg"]["block_size"])


class ActivationStore:
    """Read-only access to collected shards, as one (n, d) fp16 array (memmapped)."""

    def __init__(self, path: str | Path) -> None:
        self.dir = resolve(path)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        d = self.meta["d_model"]
        self.acts = [
            np.memmap(self.dir / f"acts_{i:03d}.bin", np.float16, "r", shape=(s, d))
            for i, s in enumerate(self.meta["shard_sizes"])
        ]
        self.tokens = [
            np.memmap(self.dir / f"tokens_{i:03d}.bin", np.uint16, "r", shape=(s,))
            for i, s in enumerate(self.meta["shard_sizes"])
        ]

    def __len__(self) -> int:
        return int(self.meta["n_tokens"])

    def load_all(self, max_tokens: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate shards into tensors (fp16 activations, int64 tokens)."""
        a = np.concatenate([np.asarray(x) for x in self.acts])
        t = np.concatenate([np.asarray(x) for x in self.tokens])
        if max_tokens is not None:
            a, t = a[:max_tokens], t[:max_tokens]
        return torch.from_numpy(a), torch.from_numpy(t.astype(np.int64))
