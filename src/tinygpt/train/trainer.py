"""Training loop: AdamW, warmup + cosine, grad accumulation and clipping, mixed precision,
periodic eval/samples/checkpoints, and exact resume."""

from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from tinygpt.data.loader import TokenLoader
from tinygpt.data.prepare import resolve, shard_paths
from tinygpt.model.generate import generate
from tinygpt.model.gpt import GPT, GPTConfig
from tinygpt.tokenizer.bpe import EOT, BPETokenizer
from tinygpt.train.metrics import device_peak_flops, mfu, pick_device
from tinygpt.train.optim import lr_at, make_optimizer
from tinygpt.utils.checkpoint import load_checkpoint, save_checkpoint
from tinygpt.utils.logging import JSONLLogger
from tinygpt.utils.run import create_run_dir
from tinygpt.utils.seed import seed_everything


@dataclass
class TrainConfig:
    """Everything that defines a training run."""

    name: str = "train"
    seed: int = 0
    model: GPTConfig = field(default_factory=GPTConfig)
    data_dir: str = "data/tinystories"
    tokenizer_path: str = "data/tokenizer/tinystories_8k.json"
    data_seed: int = 0  # data order; kept the same across model sizes
    batch_size: int = 32  # micro-batch (sequences)
    grad_accum: int = 1
    # Training length: exactly one of max_steps / train_tokens / tokens_per_param.
    max_steps: int | None = None
    train_tokens: int | None = None
    tokens_per_param: float | None = None  # D = ratio * N_nonembed (Chinchilla-style)
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int | None = None
    warmup_frac: float = 0.05  # used when warmup_steps is None
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    dtype: str = "bf16"  # "bf16" | "fp16" | "fp32"
    compile: bool = False
    device: str = "auto"
    peak_flops: float | None = None
    log_interval: int = 10
    eval_interval: int = 250
    eval_batches: int = 20
    final_eval_batches: int = 100
    eval_batch_size: int = 32
    sample_interval: int = 1000
    sample_prompt: str = "Once upon a time"
    sample_tokens: int = 96
    ckpt_interval: int = 1000
    keep_ckpt_steps: list[int] = field(default_factory=list)  # permanent snapshots
    save_final_ckpt: bool = True
    runs_root: str = "runs"
    wandb: bool = False


class Trainer:
    """Owns the model, optimiser and data loaders for one run."""

    def __init__(
        self, cfg: TrainConfig, run_dir: Path | None = None, resume_from: Path | None = None
    ) -> None:
        self.cfg = cfg
        seed_everything(cfg.seed)
        self.device = pick_device(cfg.device)
        self.model = GPT(cfg.model).to(self.device)
        self.n_params = self.model.num_params(non_embedding=True)
        self.opt = make_optimizer(self.model, cfg.lr, cfg.weight_decay, cfg.betas)
        bs, T = cfg.model.block_size, cfg.batch_size
        self.train_loader = TokenLoader(
            shard_paths(cfg.data_dir, "train"), bs, T, seed=cfg.data_seed
        )
        self.val_loader = TokenLoader(
            shard_paths(cfg.data_dir, "val"), bs, cfg.eval_batch_size, seed=0
        )
        self.tokens_per_step = cfg.batch_size * cfg.grad_accum * bs
        self.total_steps = self._total_steps()
        self.warmup = (
            cfg.warmup_steps
            if cfg.warmup_steps is not None
            else int(cfg.warmup_frac * self.total_steps)
        )
        self.step = 0
        self.flops_per_token = self.model.flops_per_token()
        self.peak = device_peak_flops(self.device, cfg.peak_flops)
        self.fwd_model: Any = torch.compile(self.model) if cfg.compile else self.model

        self.run_dir = run_dir or create_run_dir(cfg, root=resolve(cfg.runs_root), name=cfg.name)
        self.log = JSONLLogger(self.run_dir / "metrics.jsonl")
        self.samples_log = JSONLLogger(self.run_dir / "samples.jsonl")
        tok_path = resolve(cfg.tokenizer_path)
        self.tok = BPETokenizer.load(tok_path) if tok_path.exists() else None
        self._wandb = None
        if cfg.wandb:
            import wandb  # optional dependency

            self._wandb = wandb.init(project="tinygpt-lab", name=self.run_dir.name, config={})
        if resume_from is not None:
            self.load(resume_from)

    # ------------------------------------------------------------------ setup helpers
    def _total_steps(self) -> int:
        c = self.cfg
        given = [x is not None for x in (c.max_steps, c.train_tokens, c.tokens_per_param)]
        if sum(given) != 1:
            raise ValueError("set exactly one of max_steps / train_tokens / tokens_per_param")
        if c.max_steps is not None:
            return c.max_steps
        tokens = (
            c.train_tokens if c.train_tokens is not None else c.tokens_per_param * self.n_params
        )
        return max(1, math.ceil(tokens / self.tokens_per_step))

    def _autocast(self) -> contextlib.AbstractContextManager:
        if self.cfg.dtype == "fp32":
            return contextlib.nullcontext()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[self.cfg.dtype]
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    # ------------------------------------------------------------------ checkpointing
    def save_weights(self, path: Path) -> None:
        """Model weights + config only (for analysis snapshots; ~4x smaller than full)."""
        torch.save(
            {
                "model": self.model.state_dict(),
                "model_cfg": self.cfg.model.__dict__,
                "step": self.step,
                "tokens": self.step * self.tokens_per_step,
            },
            path,
        )

    def save(self, path: Path) -> None:
        save_checkpoint(
            path,
            model=self.model.state_dict(),
            optimizer=self.opt.state_dict(),
            step=self.step,
            train_loader=self.train_loader.state_dict(),
            model_cfg=self.cfg.model.__dict__,
            n_params=self.n_params,
        )

    def load(self, path: Path) -> None:
        ckpt = load_checkpoint(path, restore_rng=True)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.train_loader.load_state_dict(ckpt["train_loader"])
        self.step = int(ckpt["step"])

    # ------------------------------------------------------------------ eval & samples
    @torch.no_grad()
    def evaluate(self, n_batches: int) -> tuple[float, list[float]]:
        """Mean val loss over fixed batches, plus the per-batch losses (for bootstrap)."""
        self.model.eval()
        losses = []
        for x, y in self.val_loader.fixed_batches(n_batches, seed=1234):
            with self._autocast():
                _, loss, _ = self.fwd_model(x.to(self.device), y.to(self.device))
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses), losses

    def sample(self, n: int = 2) -> list[str]:
        if self.tok is None:
            return []
        ids = torch.tensor([self.tok.encode(self.cfg.sample_prompt)] * n, device=self.device)
        g = torch.Generator(device="cpu").manual_seed(self.cfg.seed)
        with torch.autocast(device_type=self.device.type, enabled=False):
            out = generate(
                self.model,
                ids,
                self.cfg.sample_tokens,
                temperature=0.8,
                top_k=50,
                generator=g,
                eos_id=self.tok.special_tokens.get(EOT),
            )
        self.model.train()
        return [self.tok.decode(row.tolist()) for row in out.cpu()]

    # ------------------------------------------------------------------ main loop
    def _emit(self, row: dict[str, Any]) -> None:
        self.log.log(row)
        if self._wandb is not None:
            self._wandb.log(row, step=row.get("step"))

    def train(self, stop_at: int | None = None) -> dict[str, Any]:
        """Train until `total_steps` (or `stop_at`, for tests). Returns the final metrics."""
        cfg, model = self.cfg, self.model
        model.train()
        end = self.total_steps if stop_at is None else min(stop_at, self.total_steps)
        t_last, tok_since = time.perf_counter(), 0
        t_start = time.perf_counter()
        while self.step < end:
            lr = lr_at(self.step, cfg.lr, self.warmup, self.total_steps, cfg.min_lr_ratio)
            for g in self.opt.param_groups:
                g["lr"] = lr
            loss_acc = 0.0
            for _ in range(cfg.grad_accum):
                x, y = self.train_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                with self._autocast():
                    _, loss, _ = self.fwd_model(x, y)
                (loss / cfg.grad_accum).backward()
                loss_acc += loss.item() / cfg.grad_accum
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.step += 1
            tok_since += self.tokens_per_step
            if not math.isfinite(loss_acc):
                self._emit({"step": self.step, "train_loss": loss_acc, "diverged": True})
                raise FloatingPointError(f"loss became {loss_acc} at step {self.step}")

            if self.step % cfg.log_interval == 0 or self.step == end:
                if self.device.type == "mps":
                    torch.mps.synchronize()
                elif self.device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t_last
                tps = tok_since / dt
                self._emit(
                    {
                        "step": self.step,
                        "tokens": self.step * self.tokens_per_step,
                        "train_loss": loss_acc,
                        "lr": lr,
                        "grad_norm": gnorm,
                        "tokens_per_sec": tps,
                        "mfu": mfu(self.flops_per_token, tps, self.peak),
                    }
                )
                t_last, tok_since = time.perf_counter(), 0
            if self.step % cfg.eval_interval == 0 and self.step != self.total_steps:
                vl, _ = self.evaluate(cfg.eval_batches)
                self._emit({"step": self.step, "val_loss": vl})
            if self.step % cfg.sample_interval == 0:
                self.samples_log.log({"step": self.step, "samples": self.sample()})
            if self.step % cfg.ckpt_interval == 0 and self.step != self.total_steps:
                self.save(self.run_dir / "ckpt_last.pt")
            if self.step in cfg.keep_ckpt_steps:
                self.save_weights(self.run_dir / f"weights_step{self.step:06d}.pt")

        result: dict[str, Any] = {"step": self.step}
        if self.step == self.total_steps:
            result = self.finish(time.perf_counter() - t_start)
        return result

    def finish(self, wall_s: float) -> dict[str, Any]:
        vl, per_batch = self.evaluate(self.cfg.final_eval_batches)
        tokens = self.step * self.tokens_per_step
        result = {
            "step": self.step,
            "val_loss": vl,
            "val_loss_per_batch": per_batch,
            "n_params_nonembed": self.n_params,
            "n_params_total": self.model.num_params(non_embedding=False),
            "tokens": tokens,
            "compute_6ND": 6.0 * self.n_params * tokens,
            "compute_flops": self.flops_per_token * tokens,
            "wall_seconds_this_session": wall_s,
            "device": str(self.device),
        }
        self._emit({"step": self.step, "val_loss": vl, "final": True})
        (self.run_dir / "final.json").write_text(json.dumps(result, indent=2))
        if self.cfg.save_final_ckpt:
            self.save(self.run_dir / "ckpt_final.pt")
        self.samples_log.log({"step": self.step, "samples": self.sample(4)})
        return result

    def close(self) -> None:
        self.log.close()
        self.samples_log.close()
        if self._wandb is not None:
            self._wandb.finish()
