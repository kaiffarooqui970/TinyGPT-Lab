"""Run-directory creation: every run stores its config, git hash and environment."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from tinygpt.utils.config import save_config

REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_hash(cwd: Path = REPO_ROOT) -> str:
    """Full commit hash of HEAD, or 'nogit' if unavailable (e.g. no commits yet)."""
    return _git(["rev-parse", "HEAD"], cwd) or "nogit"


def git_dirty(cwd: Path = REPO_ROOT) -> bool:
    """True if tracked files have uncommitted changes, which makes the hash not exact."""
    status = _git(["status", "--porcelain", "--untracked-files=no"], cwd)
    return bool(status)


def create_run_dir(cfg: Any, root: str | Path = REPO_ROOT / "runs", name: str = "run") -> Path:
    """Create `root/<timestamp>_<name>/` holding `config.yaml` and `meta.json`.

    Args:
        cfg: The dataclass config driving this run.
        root: Parent directory for runs (gitignored).
        name: Human-readable run name.

    Returns:
        The path of the new run directory.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = root / f"{stamp}_{name}"
    run_dir.mkdir()  # fails if it already exists, so runs are never overwritten
    save_config(cfg, run_dir / "config.yaml")
    meta = {
        "name": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "git_dirty": git_dirty(),
        "argv": sys.argv,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir
