"""Append-only JSONL metric logger: one JSON object per line, flushed after each write."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import torch


def _to_py(v: Any) -> Any:
    """Convert tensors and NumPy scalars to plain Python values for JSON."""
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)  # JSON has no NaN/inf; keep the value visible instead of dropping it
    return v


class JSONLLogger:
    """Writes metric dicts to a `.jsonl` file. Flushing every line makes a crashed run
    keep everything logged up to that point."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a")  # noqa: SIM115 (long-lived handle, closed in close())

    def log(self, row: dict[str, Any]) -> None:
        """Append one row."""
        self._fh.write(json.dumps({k: _to_py(v) for k, v in row.items()}) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> JSONLLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read all rows from a JSONL file."""
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]
