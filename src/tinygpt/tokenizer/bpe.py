"""Byte-level BPE tokenizer, written from scratch.

Pipeline: text -> pre-tokens (GPT-2-style regex split) -> UTF-8 bytes -> merges.

* Vocabulary: ids 0..255 are raw bytes, then one id per learned merge, then special
  tokens. Every string is representable, so there is no UNK token, and decode(encode(x))
  == x exactly.
* Merges never cross pre-token boundaries, so "dog." and "dog!" share the token "dog".
* Training is incremental. We keep global pair counts and an index from each pair to
  the words containing it. Applying a merge touches only the words that contain the
  merged pair, so we never recount the whole corpus. A max-heap with lazy invalidation
  finds the next best pair.
* Ties are broken by the smallest (left_id, right_id), which makes training fully
  deterministic.
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

EOT = "<|endoftext|>"

# GPT-2's split pattern, rewritten for Python's `re` (no \p{L}): `[^\W\d_]` = letters.
# The final `[\s\S]` alternative matches any leftover char, so the split is lossless.
PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+|[\s\S]"""
)

Pair = tuple[int, int]


def pretokenize(text: str) -> list[str]:
    """Split text into pre-tokens; ''.join(pretokenize(t)) == t."""
    return PRETOKEN_PATTERN.findall(text)


def _count_words(texts: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for t in texts:
        counts.update(pretokenize(t))
    return counts


def _merge_word(word: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace non-overlapping occurrences of `pair` (left to right) with `new_id`."""
    a, b = pair
    out: list[int] = []
    i, n = 0, len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def _pairs(word: list[int]) -> list[Pair]:
    return list(zip(word, word[1:], strict=False))


class BPETokenizer:
    """Byte-level BPE tokenizer with special tokens."""

    def __init__(self, merges: list[Pair], special_tokens: dict[str, int] | None = None) -> None:
        self.merges: list[Pair] = [tuple(m) for m in merges]  # type: ignore[misc]
        self.ranks: dict[Pair, int] = {p: i for i, p in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.special_tokens: dict[str, int] = dict(special_tokens or {})
        for s, i in self.special_tokens.items():
            self.vocab[i] = s.encode("utf-8")
        self._special_re = (
            re.compile(
                "("
                + "|".join(map(re.escape, sorted(self.special_tokens, key=len, reverse=True)))
                + ")"
            )
            if self.special_tokens
            else None
        )
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ training
    @classmethod
    def train(
        cls, texts: Iterable[str], vocab_size: int = 8192, special_tokens: list[str] | None = None
    ) -> BPETokenizer:
        """Learn merges from `texts` until the vocabulary has `vocab_size` entries.

        Args:
            texts: Training documents. Special tokens should not appear in them.
            vocab_size: Total size: 256 bytes + merges + special tokens.
            special_tokens: Strings that get reserved ids at the end of the vocabulary.
        """
        special_tokens = special_tokens or []
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError("vocab_size too small for bytes + special tokens")

        word_counts = _count_words(texts)
        words: list[list[int]] = [list(w.encode("utf-8")) for w in word_counts]
        freqs: list[int] = list(word_counts.values())

        pair_counts: dict[Pair, int] = {}
        where: dict[Pair, set[int]] = {}
        for wi, (w, f) in enumerate(zip(words, freqs, strict=True)):
            for p in _pairs(w):
                pair_counts[p] = pair_counts.get(p, 0) + f
                where.setdefault(p, set()).add(wi)

        heap: list[tuple[int, Pair]] = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[Pair] = []
        while len(merges) < n_merges and heap:
            neg, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) != -neg or -neg <= 0:
                continue  # stale heap entry
            new_id = 256 + len(merges)
            merges.append(pair)
            changed: set[Pair] = set()
            for wi in where.pop(pair, set()):
                old = words[wi]
                new = _merge_word(old, pair, new_id)
                if len(new) == len(old):
                    continue  # lazy index: this word no longer contains the pair
                f = freqs[wi]
                for p in _pairs(old):
                    pair_counts[p] -= f
                    changed.add(p)
                for p in _pairs(new):
                    pair_counts[p] = pair_counts.get(p, 0) + f
                    where.setdefault(p, set()).add(wi)
                    changed.add(p)
                words[wi] = new
            pair_counts.pop(pair, None)
            for p in changed:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)

        specials = {s: 256 + len(merges) + i for i, s in enumerate(special_tokens)}
        return cls(merges, specials)

    @classmethod
    def train_naive(cls, texts: Iterable[str], vocab_size: int) -> BPETokenizer:
        """Reference trainer that recounts all pairs every merge. Used only in tests."""
        word_counts = _count_words(texts)
        words = [list(w.encode("utf-8")) for w in word_counts]
        freqs = list(word_counts.values())
        merges: list[Pair] = []
        while len(merges) < vocab_size - 256:
            counts: Counter[Pair] = Counter()
            for w, f in zip(words, freqs, strict=True):
                for p in _pairs(w):
                    counts[p] += f
            if not counts:
                break
            pair = min(counts, key=lambda p: (-counts[p], p))
            new_id = 256 + len(merges)
            merges.append(pair)
            words = [_merge_word(w, pair, new_id) for w in words]
        return cls(merges)

    # ------------------------------------------------------------------ encode/decode
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        word = list(chunk.encode("utf-8"))
        while len(word) > 1:
            best = min(_pairs(word), key=lambda p: self.ranks.get(p, 1 << 62))
            rank = self.ranks.get(best)
            if rank is None:
                break
            word = _merge_word(word, best, 256 + rank)
        if len(self._cache) < 1_000_000:
            self._cache[chunk] = word
        return word

    def encode(self, text: str, allowed_special: bool = False) -> list[int]:
        """Encode text to ids. When `allowed_special` is set, special-token strings map to their
        reserved ids; otherwise they are encoded as ordinary text."""
        parts = self._special_re.split(text) if (allowed_special and self._special_re) else [text]
        ids: list[int] = []
        for part in parts:
            if not part:
                continue
            if allowed_special and part in self.special_tokens:
                ids.append(self.special_tokens[part])
                continue
            for chunk in pretokenize(part):
                ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        """Decode ids to text. Invalid UTF-8 (a truncated multi-byte char) becomes U+FFFD."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors=errors)

    def bytes_per_token(self, texts: Iterable[str]) -> float:
        """Compression ratio: UTF-8 bytes / tokens over `texts`."""
        n_bytes = n_tok = 0
        for t in texts:
            n_bytes += len(t.encode("utf-8"))
            n_tok += len(self.encode(t))
        return n_bytes / max(n_tok, 1)

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "type": "byte_bpe",
                    "merges": self.merges,
                    "special_tokens": self.special_tokens,
                }
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        d = json.loads(Path(path).read_text())
        return cls([tuple(m) for m in d["merges"]], d["special_tokens"])
