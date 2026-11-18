"""Phase 1 acceptance tests for the byte-level BPE tokenizer."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from tinygpt.tokenizer.bpe import EOT, BPETokenizer, pretokenize

CORPUS = [
    "Once upon a time, there was a little girl named Lily. She loved to play outside.",
    "One day, Tom saw a big red ball in the park. He wanted to play with it!",
    "The cat sat on the mat. The dog barked at the cat, and the cat ran away.",
    "Lily and Tom were friends. They liked to share their toys and sing songs.",
    "Café naïve — 日本語のテキスト 😀 emojis 🎉 and numbers 12345 too.",
] * 20


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=400, special_tokens=[EOT])


def _random_unicode(rng: random.Random, n: int) -> str:
    ranges = [
        (0x20, 0x7E),
        (0xA0, 0x24F),
        (0x370, 0x3FF),
        (0x400, 0x4FF),
        (0x3040, 0x30FF),
        (0x4E00, 0x4FFF),
        (0x1F300, 0x1F64F),
    ]
    out = []
    for _ in range(n):
        lo, hi = rng.choice(ranges)
        out.append(chr(rng.randint(lo, hi)))
        if rng.random() < 0.15:
            out.append(rng.choice([" ", "\n", "\t", "  ", "'s", "'ll"]))
    return "".join(out)


def test_pretokenize_lossless() -> None:
    s = "Hello  world!\n\nIt's 2024 — 日本語  ok? "
    assert "".join(pretokenize(s)) == s


def test_roundtrip_random_unicode(tok: BPETokenizer) -> None:
    rng = random.Random(0)
    for _ in range(200):
        s = _random_unicode(rng, rng.randint(0, 80))
        assert tok.decode(tok.encode(s)) == s


def test_roundtrip_corpus(tok: BPETokenizer) -> None:
    for s in CORPUS[:5]:
        ids = tok.encode(s)
        assert tok.decode(ids) == s
        assert len(ids) < len(s.encode("utf-8"))  # merges actually compress


def test_vocab_size_exact(tok: BPETokenizer) -> None:
    assert tok.vocab_size == 400
    assert max(tok.encode("".join(CORPUS[:5]))) < tok.vocab_size


def test_merge_determinism() -> None:
    a = BPETokenizer.train(CORPUS, vocab_size=350, special_tokens=[EOT])
    b = BPETokenizer.train(list(CORPUS), vocab_size=350, special_tokens=[EOT])
    assert a.merges == b.merges


def test_incremental_matches_naive() -> None:
    """The incremental pair-count trainer must pick the same merges as a full recount."""
    fast = BPETokenizer.train(CORPUS, vocab_size=320, special_tokens=[])
    slow = BPETokenizer.train_naive(CORPUS, vocab_size=320)
    assert fast.merges == slow.merges


def test_special_tokens(tok: BPETokenizer) -> None:
    eot = tok.special_tokens[EOT]
    ids = tok.encode(f"Hello{EOT}World", allowed_special=True)
    assert ids.count(eot) == 1
    assert tok.decode(ids) == f"Hello{EOT}World"
    # Without allowed_special the string is treated as plain text, never as the special id.
    plain = tok.encode(f"Hello{EOT}World", allowed_special=False)
    assert eot not in plain
    assert tok.decode(plain) == f"Hello{EOT}World"


def test_save_load(tok: BPETokenizer, tmp_path: Path) -> None:
    path = tmp_path / "tok.json"
    tok.save(path)
    tok2 = BPETokenizer.load(path)
    s = "Lily saw a 🎉 big cat."
    assert tok2.encode(s) == tok.encode(s)
    assert tok2.merges == tok.merges and tok2.special_tokens == tok.special_tokens


def test_bytes_per_token(tok: BPETokenizer) -> None:
    bpt = tok.bytes_per_token(CORPUS[:5])
    assert bpt > 1.5
