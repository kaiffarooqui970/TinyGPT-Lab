# Design decisions

One line of reasoning per decision. Newest at the bottom of each phase.

## Phase 0 — Setup

- **Python 3.12 (pinned in `.python-version`)**: the newest version with mature PyTorch wheels on all platforms; the system has 3.15, which torch support lags behind.
- **Repo at `~/tinygpt-lab`, not on the Desktop**: iCloud Desktop sync evicts venv files, which makes imports hang.
- **src layout + hatchling editable install**: stops tests from importing the working directory by accident, so they exercise the installed package.
- **Extra modules `utils/config.py` and `utils/run.py`** (not in the PLAN layout): config validation and run-dir creation are separate concerns from seeding and logging; keeping each module small follows the "small focused modules" rule.
- **Configs are dataclasses validated by a small hand-written `from_dict`**: unknown keys, missing fields and wrong types fail at load time, with no pydantic dependency (not on the allowed list).
- **`bool` is rejected where an `int` is expected; `int` and numeric strings are accepted for `float`**: YAML parses `1e-3` as a string and `1` as an int, and silently treating `True` as 1 hides typos.
- **Run dir = `runs/<YYYYmmdd-HHMMSS-micro>_<name>/` with `config.yaml`, `meta.json` (git hash, dirty flag, argv, versions), `metrics.jsonl`**: everything needed to trace a number back to code + config; `mkdir` without `exist_ok`, so runs are never overwritten.
- **`git_dirty` recorded alongside the hash**: a hash alone is misleading if the run used uncommitted code.
- **JSONL logger flushes every line; NaN/inf logged as strings**: crashed runs keep their logs; JSON has no NaN, and dropping a diverged value would hide the divergence.
- **`seed_everything(deterministic=False)` by default**: deterministic kernels slow CUDA training; they are switched on only for the bit-exact resume tests.
- **wandb is an optional extra (`uv sync --extra wandb`)**: the JSONL logs are the source of truth for every reported number.
- **pre-commit runs ruff + ruff-format + pytest**: lint and the test suite gate every commit.
- **pre-commit hooks call `uv run ruff`/`uv run pytest` (local hooks)**: the hook versions then match `uv.lock` exactly; a separately pinned ruff disagreed on deprecated rules.

## Run mode

- **All phases run without stopping between them** (the user's instruction on 2026-11-18, overriding PLAN's per-phase gate). Every phase still ends with tests and a commit.
- **Compute: Apple M4 (MPS), not a 24 GB CUDA GPU**: this machine has no CUDA, so experiments are scaled down to fit hours of M4 time and ~9 GB of free disk. All sizes stay configurable; the README gives the GPU-scale configs and estimates.

## Phase 1 — Tokenizer

- **GPT-2 pre-tokenization regex rewritten for stdlib `re`** (`[^\W\d_]` for letters, plus a `[\s\S]` catch-all): the `regex` package is not on the allowed list, and the catch-all keeps the split lossless.
- **Merge ties broken by the smallest (left, right) id pair**: makes training deterministic regardless of dict or heap ordering; tested against a naive full-recount trainer.
- **Incremental training: pair counts + pair→words index + lazy max-heap**: each merge touches only the words containing that pair (7,935 merges over 180 MB in ~7 s).
- **Special tokens get the last ids; `encode` treats them as text unless `allowed_special=True`**: user text containing "<|endoftext|>" can never inject a document boundary.
- **Tokenizer trained on the first 200k TinyStories docs (~180 MB)**: merge statistics saturate long before the full 1.9 GB, and held-out compression (4.16 B/tok) matches train (4.14 B/tok).

## Phase 2 — Data

- **Every document is followed by `<|endoftext|>`, and windows may span documents**: standard GPT pre-training packing; the model learns to reset context at EOT, and no tokens are lost to padding.
- **Windows never cross shard boundaries; shards are sampled in proportion to their valid start positions**: every start position across the dataset is equally likely, and no seams are introduced.
- **Random windows (sampled with replacement) from a single seeded `torch.Generator`**: the data order is fixed by the seed, identical across model sizes, and checkpointable (generator state + step).
- **Eval uses `fixed_batches(seed)`, a separate generator**: the val loss is always measured on exactly the same windows, unaffected by training draws.
- **Tokenization runs in a spawn-based process pool with `imap` (ordered)**: the output is identical to a single-process run.
- TinyStories: 466,769,099 train tokens, 4,691,376 val tokens (8192-vocab BPE).

## Phase 3 — Model

- **RoPE in 'rotate-half' layout, applied to q and k after the projection, with a position offset for the KV cache**: standard (GPT-NeoX/LLaMA); relative-position attention with no learned position parameters.
- **One fused qkv linear; the attention softmax is computed in fp32 in the manual path**: fewer kernels; fp32 softmax avoids bf16 overflow.
- **Cached attention with T > 1 uses an explicit offset mask `tril(diagonal=S-T)`**: SDPA's `is_causal` assumes square masks and would be wrong for chunked prefill.
- **SwiGLU hidden = 8/3·d rounded up to a multiple of 64**: same parameter count as a 4·d GELU MLP; multiples of 64 suit matmul kernels.
- **Init N(0, 0.02); residual output projections (attn.proj, mlp.w2) N(0, 0.02/√(2L)); RMSNorm gains = 1**: GPT-2 scheme, which keeps the residual stream variance roughly constant with depth.
- **Non-embedding params N exclude the tied embedding matrix (Kaplan et al. 2020)**: power laws in N are cleaner without embeddings, which matter a lot at small scale.
- **FLOPs/token = 6N + 12·L·d·T (+ 6·d·V for the lm head)**: PaLM's accounting, used for MFU; the scaling study's compute axis uses the standard C = 6·N·D.
- **Generation with cache crops the prompt to block_size−1 and restarts the cache when the context fills**: RoPE positions never exceed the table length; recomputing on overflow keeps it simple and correct.

## Phase 4 — Training

- **LR schedule is a pure function of the step (`lr_at`), not a torch scheduler object**: resume needs no scheduler state, and the schedule is trivially testable.
- **No weight decay on norm gains or the (tied) embedding; wd = 0.1, betas (0.9, 0.95), clip 1.0, warmup 5% then cosine to 0.1·lr**: GPT-3/Chinchilla-style defaults.
- **Resume restores model, optimiser, step, loader generator state and the Python/NumPy/torch(/CUDA/MPS) RNGs**; the test compares 8 straight steps with 3 + save + scramble RNG + resume + 5, bit-for-bit (CPU, fp32).
- **bf16 autocast on MPS** (measured ~5% faster than fp32 at these sizes; fp32 is used in CPU tests for exactness).
- **Weights-only snapshots (`weights_step*.pt`) at log-spaced steps**: ~4× smaller than full checkpoints (no Adam state); enough for analysis such as SAEs on intermediate checkpoints.
- **The final eval logs all 200 per-batch val losses** (`final.json`): the scaling fit's bootstrap CIs resample them.
- **`final.json` and the checkpoint are written before the final samples are generated**: a sampling bug once crashed a sweep run after training had finished, which lost the run; results now persist first.
- **Long runs are wrapped in `caffeinate -ims`**: macOS was idle-throttling a background run (12 s of CPU in 17 min).
- **The ~10M Phase 4 acceptance model is the same run as the largest scaling point (S5)**: on M4 throughput (~14.6k tok/s), one 212M-token run costs ~4 h, so training it twice would double the dominant cost.

## Phase 5 — Scaling

- **Sizes S1–S5 = 0.43M, 0.95M, 1.77M, 4.82M, 10.6M non-embedding params** (not 1M–50M): measured M4 throughput (7.7k tok/s at 25M) puts a 25M model at ~18 h, and 50M × 20 tok/param = 1B tokens exceeds TinyStories (467M). The range still spans 25× in N.
- **Shapes: d_model/n_layer between 53 and 64, head_dim 32 (S1–S4) or 64 (S5)**: Kaplan et al. show loss depends weakly on shape at fixed N, so we vary width and depth together.
- **Constant batch of 8,192 tokens for all sizes**: a simplification; the critical batch size grows as loss falls, so the larger models may be slightly under-batched (noted as a limitation).
- **Same data order (data_seed 0) for every size**: differences between sizes are not data-order noise.
- **LR sweep at S1: 7 log-spaced LRs; optimum 4.8e-3 (val 2.608), with a clear U-shape (3.55 at 3e-4, 3.09 at 1.9e-2)**.
- **LR transfer: Kaplan et al. (2020) eq. D.1, LR(N) ∝ 0.003239 − 0.0001395·ln N, anchored at the S1 optimum** → 4.8e-3, 4.4e-3, 4.1e-3, 3.7e-3, 3.3e-3. A simple, published rule instead of µP (which would need a re-parameterised model; out of time budget). Checked by a 0.5×/2× sweep at S3.
- **Bootstrap CIs resample the 200 per-batch val losses of each run** (eval noise only; seed-to-seed variance would need repeated runs, which we cannot afford — stated as a limitation).

## Phase 9 — Extension choice (made without waiting, per the user's instruction)

Options considered: (1) universality across scale, (2) feature emergence over training (S5 weight snapshots), (3) SAE quality vs. model size.
- **Chosen: (1) combined with (3)**: SAEs on S3/S4/S5 at relative depth ½ on *identical tokens*, matched by activation correlation (decoder cosine is undefined across different d_model). A second S5 SAE with another seed gives the seed-noise ceiling, so "absent at small scale" can be separated from "the SAE is not reproducible". Fidelity, dead latents and L0 at a fixed expansion factor give (3) for free. Option (2) remains possible because S5 snapshots are kept.
- **LR-transfer check at S3 passed**: 0.5× → 2.0135, 1× → 1.9974, 2× → 2.0035 val loss. The transferred LR is best of the three, so the Kaplan rule is well calibrated at least up to S3 (the optimum is fairly flat: ±0.016 nats over a 4× LR range).
- **Ops: the Mac slept for ~8.5 h overnight** (battery, lid closed; `caffeinate` cannot prevent that). Runs resume automatically on wake. The logged tokens/s for the interval spanning a sleep (and `wall_seconds_this_session` of that run) include the sleep, so they understate throughput. Long runs need AC power.

## Phase 6 — Activations

- **Activations come from the held-out val split as sequential non-overlapping 256-token windows, 2,097,152 tokens per model (S3, S4, S5)**: free disk fell to ~4 GB during training (macOS swap growth), so the planned 4M-token S5 store (+2M each for S3/S4, ~5 GB) would not fit. Sequential windows use every token exactly once, and the tokens are identical across models, as Phase 9 needs. The PLAN's 50–100M tokens is ~25–50× more than we can store here.
- **SAE CE evaluation and interventions sample only val tokens after the collected region (`min_start`)**: the downstream metrics never use tokens whose activations the SAE trained on.
