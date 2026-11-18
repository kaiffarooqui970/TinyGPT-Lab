"""Dataset download, tokenizer training input, and tokenization into uint16 memmap shards.

Layout of a prepared dataset directory:

    out_dir/
      train_000.bin, train_001.bin, ...   # flat uint16 token ids, docs separated by EOT
      val_000.bin
      meta.json                           # token counts, tokenizer path, source info

Documents keep their original dataset order, so the data order is deterministic.
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tinygpt.tokenizer.bpe import EOT, BPETokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
HF_HOME = REPO_ROOT / "data" / "hf"


@dataclass
class DatasetConfig:
    """Which text corpus to use."""

    hf_name: str = "roneneldan/TinyStories"
    hf_config: str | None = None
    text_field: str = "text"
    train_split: str = "train"
    val_split: str | None = "validation"  # None: hold out the last `val_docs` train docs
    val_docs: int = 20_000
    max_train_docs: int | None = None
    streaming: bool = False  # use for large corpora (FineWeb-Edu sample)


@dataclass
class TokenizerConfig:
    """Config for `scripts/train_tokenizer.py`."""

    dataset: DatasetConfig
    vocab_size: int = 8192
    train_docs: int = 200_000  # first N train docs are used to learn merges
    heldout_docs: int = 5_000  # first N val docs are used for the compression ratio
    out_path: str = "data/tokenizer/tinystories_8k.json"
    seed: int = 0


@dataclass
class PrepareConfig:
    """Config for `scripts/prepare_data.py`."""

    dataset: DatasetConfig
    tokenizer_path: str = "data/tokenizer/tinystories_8k.json"
    out_dir: str = "data/tinystories"
    shard_tokens: int = 100_000_000
    num_workers: int = 8
    chunk_docs: int = 2_000
    seed: int = 0


def resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def iter_texts(cfg: DatasetConfig, split: str, limit: int | None = None) -> Iterator[str]:
    """Yield documents of `split` ('train' or 'val') in dataset order."""
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    from datasets import load_dataset  # imported lazily: slow import, download-only use

    if split == "val" and cfg.val_split is not None:
        ds = load_dataset(cfg.hf_name, cfg.hf_config, split=cfg.val_split, streaming=cfg.streaming)
        it: Iterator[str] = (r[cfg.text_field] for r in ds)
    else:
        ds = load_dataset(
            cfg.hf_name, cfg.hf_config, split=cfg.train_split, streaming=cfg.streaming
        )
        it = (r[cfg.text_field] for r in ds)
        if cfg.val_split is None:
            # The first `val_docs` documents form the val set; train starts after them.
            it = (
                itertools.islice(it, cfg.val_docs)
                if split == "val"
                else itertools.islice(it, cfg.val_docs, None)
            )
        if split == "train" and cfg.max_train_docs is not None:
            it = itertools.islice(it, cfg.max_train_docs)
    if limit is not None:
        it = itertools.islice(it, limit)
    yield from it


# ---------------------------------------------------------------------- tokenization
_TOK: BPETokenizer | None = None


def _init_worker(tok_path: str) -> None:
    global _TOK
    _TOK = BPETokenizer.load(tok_path)


def _encode_docs(docs: list[str]) -> np.ndarray:
    assert _TOK is not None
    eot = _TOK.special_tokens[EOT]
    ids: list[int] = []
    for d in docs:
        ids.extend(_TOK.encode(d))
        ids.append(eot)  # EOT closes every doc, so the model learns where docs end
    return np.asarray(ids, dtype=np.uint16)


def _chunks(it: Iterator[str], n: int) -> Iterator[list[str]]:
    while chunk := list(itertools.islice(it, n)):
        yield chunk


class ShardWriter:
    """Streams uint16 tokens into fixed-size shard files."""

    def __init__(self, out_dir: Path, prefix: str, shard_tokens: int) -> None:
        self.out_dir, self.prefix, self.shard_tokens = out_dir, prefix, shard_tokens
        self.shard_idx, self.in_shard, self.total = 0, 0, 0
        self._fh = None

    def write(self, arr: np.ndarray) -> None:
        while arr.size:
            if self._fh is None:
                path = self.out_dir / f"{self.prefix}_{self.shard_idx:03d}.bin"
                self._fh = open(path, "wb")  # noqa: SIM115
            take = min(arr.size, self.shard_tokens - self.in_shard)
            arr[:take].tofile(self._fh)
            self.in_shard += take
            self.total += take
            arr = arr[take:]
            if self.in_shard == self.shard_tokens:
                self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self.shard_idx += 1
            self.in_shard = 0


def tokenize_split(cfg: PrepareConfig, split: str) -> int:
    """Tokenize one split into shards and return the number of tokens written."""
    out_dir = resolve(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{split}_*.bin"):
        old.unlink()
    tok_path = str(resolve(cfg.tokenizer_path))
    tok = BPETokenizer.load(tok_path)
    if tok.vocab_size > 2**16:
        raise ValueError("uint16 shards need vocab_size <= 65536")
    writer = ShardWriter(out_dir, split, cfg.shard_tokens)
    docs = _chunks(iter_texts(cfg.dataset, split), cfg.chunk_docs)
    ctx = mp.get_context("spawn")
    with ctx.Pool(cfg.num_workers, initializer=_init_worker, initargs=(tok_path,)) as pool:
        # imap (not imap_unordered) keeps document order deterministic.
        for i, arr in enumerate(pool.imap(_encode_docs, docs, chunksize=1)):
            writer.write(arr)
            if i % 100 == 0:
                print(f"[{split}] chunk {i}  tokens so far {writer.total:,}", flush=True)
    writer.close()
    return writer.total


def shard_paths(data_dir: str | Path, split: str) -> list[Path]:
    """Sorted shard files of a split."""
    paths = sorted(resolve(data_dir).glob(f"{split}_*.bin"))
    if not paths:
        raise FileNotFoundError(f"no {split} shards in {data_dir}")
    return paths


def write_meta(cfg: PrepareConfig, counts: dict[str, int]) -> None:
    meta = {
        "tokens": counts,
        "tokenizer_path": cfg.tokenizer_path,
        "dataset": cfg.dataset.hf_name,
        "dtype": "uint16",
    }
    (resolve(cfg.out_dir) / "meta.json").write_text(json.dumps(meta, indent=2))
