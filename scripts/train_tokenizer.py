"""Train the byte-level BPE tokenizer and report its compression on held-out text.

Usage:
    uv run python scripts/train_tokenizer.py --config configs/tokenizer.yaml
"""

from __future__ import annotations

import argparse
import time

from tinygpt.data.prepare import TokenizerConfig, iter_texts, resolve
from tinygpt.tokenizer.bpe import EOT, BPETokenizer
from tinygpt.utils.config import load_config
from tinygpt.utils.logging import JSONLLogger
from tinygpt.utils.run import create_run_dir
from tinygpt.utils.seed import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config, TokenizerConfig)
    seed_everything(cfg.seed)
    run_dir = create_run_dir(cfg, name="tokenizer")

    texts = list(iter_texts(cfg.dataset, "train", limit=cfg.train_docs))
    t0 = time.time()
    tok = BPETokenizer.train(texts, vocab_size=cfg.vocab_size, special_tokens=[EOT])
    train_s = time.time() - t0
    out = resolve(cfg.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(out)
    tok.save(run_dir / "tokenizer.json")

    heldout = list(iter_texts(cfg.dataset, "val", limit=cfg.heldout_docs))
    bpt_heldout = tok.bytes_per_token(heldout)
    bpt_train = tok.bytes_per_token(texts[: cfg.heldout_docs])
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    result = {
        "vocab_size": tok.vocab_size,
        "n_merges": len(tok.merges),
        "train_docs": len(texts),
        "train_bytes": n_bytes,
        "train_seconds": round(train_s, 2),
        "bytes_per_token_heldout": bpt_heldout,
        "bytes_per_token_train_sample": bpt_train,
        "heldout_docs": len(heldout),
    }
    with JSONLLogger(run_dir / "metrics.jsonl") as log:
        log.log(result)
    longest = sorted(tok.vocab.values(), key=len, reverse=True)[:15]
    print(result)
    print("longest tokens:", [t.decode("utf-8", "replace") for t in longest])
    print(run_dir)


if __name__ == "__main__":
    main()
