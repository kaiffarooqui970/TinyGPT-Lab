"""Phase 0 acceptance tests: seeding, config loading, run dirs, JSONL logging."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch

from tinygpt.utils.config import ConfigError, from_dict, load_config
from tinygpt.utils.logging import JSONLLogger, read_jsonl
from tinygpt.utils.run import create_run_dir, git_hash
from tinygpt.utils.seed import seed_everything

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- seeding
def _draw() -> tuple[float, float, list[float]]:
    return random.random(), float(np.random.rand()), torch.rand(3).tolist()


def test_seed_everything_is_reproducible() -> None:
    seed_everything(123)
    a = _draw()
    seed_everything(123)
    b = _draw()
    assert a == b


def test_different_seeds_differ() -> None:
    seed_everything(1)
    a = _draw()
    seed_everything(2)
    assert a != _draw()


# ---------------------------------------------------------------- config
@dataclass
class Inner:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.95)


@dataclass
class Outer:
    name: str
    steps: int = 10
    tags: list[str] = field(default_factory=list)
    optim: Inner = field(default_factory=Inner)
    note: str | None = None


def test_from_dict_nested_and_defaults() -> None:
    cfg = from_dict(Outer, {"name": "x", "optim": {"lr": 3e-4}})
    assert cfg.name == "x" and cfg.steps == 10
    assert cfg.optim.lr == 3e-4 and cfg.optim.betas == (0.9, 0.95)


def test_from_dict_rejects_unknown_key() -> None:
    with pytest.raises(ConfigError, match="unknown"):
        from_dict(Outer, {"name": "x", "stpes": 3})


def test_from_dict_rejects_missing_required() -> None:
    with pytest.raises(ConfigError, match="missing"):
        from_dict(Outer, {})


def test_from_dict_rejects_wrong_type() -> None:
    with pytest.raises(ConfigError, match="steps"):
        from_dict(Outer, {"name": "x", "steps": "ten"})
    with pytest.raises(ConfigError):  # bool is not a valid int in configs
        from_dict(Outer, {"name": "x", "steps": True})


def test_int_promotes_to_float() -> None:
    assert from_dict(Inner, {"lr": 1}).lr == 1.0


def test_load_config_yaml(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("name: demo\nsteps: 5\noptim:\n  lr: 1.0e-4\n")
    cfg = load_config(p, Outer)
    assert cfg.steps == 5 and cfg.optim.lr == 1e-4


# ---------------------------------------------------------------- run dir
def test_create_run_dir_saves_config_and_meta(tmp_path: Path) -> None:
    cfg = Outer(name="demo")
    run = create_run_dir(cfg, root=tmp_path, name="demo")
    assert run.exists() and run.parent == tmp_path
    saved = (run / "config.yaml").read_text()
    assert "name: demo" in saved
    meta = json.loads((run / "meta.json").read_text())
    assert {"git_hash", "git_dirty", "timestamp", "python", "torch"} <= meta.keys()


def test_run_dirs_are_unique(tmp_path: Path) -> None:
    cfg = Outer(name="demo")
    a = create_run_dir(cfg, root=tmp_path, name="demo")
    b = create_run_dir(cfg, root=tmp_path, name="demo")
    assert a != b


def test_git_hash_in_repo() -> None:
    h = git_hash(REPO)
    assert h == "nogit" or len(h) == 40


# ---------------------------------------------------------------- logger
def test_jsonl_logger_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    with JSONLLogger(path) as log:
        log.log({"step": 0, "loss": 1.5})
        log.log({"step": 1, "loss": torch.tensor(1.25), "lr": np.float32(0.5)})
    rows = read_jsonl(path)
    assert rows == [{"step": 0, "loss": 1.5}, {"step": 1, "loss": 1.25, "lr": 0.5}]


# ---------------------------------------------------------------- hello_run
def test_hello_run_script(tmp_path: Path) -> None:
    out = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/hello_run.py"),
            "--config",
            str(REPO / "configs/hello.yaml"),
            "--runs-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    run_dir = Path(out.stdout.strip().splitlines()[-1])
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "meta.json").exists()
    assert len(read_jsonl(run_dir / "metrics.jsonl")) > 0
