# PLAN — tinygpt-lab (source of truth)

You are my research engineer for this project. First, save this whole prompt to PLAN.md in the repo root and treat it as the source of truth. Work phase by phase. At the end of every phase: run the tests, show me the key results (numbers and plots), commit with a clear message, and stop and wait for my go-ahead before starting the next phase.

## Goal

Build a research-grade, reproducible project that shows real ML depth:

- A GPT-style language model implemented from scratch in plain PyTorch.
- A scaling-law study across 4–5 model sizes with fitted power laws.
- A sparse autoencoder (SAE) trained on the model's internal activations to find interpretable features, plus causal tests of those features.
- One original research extension.
- A paper-style write-up.

## Hard rules

- No high-level model libraries. Do not use HuggingFace transformers, nanoGPT code, TransformerLens, or sae_lens. Allowed: torch, numpy, scipy, matplotlib, pyyaml, tqdm, pytest, datasets (download only), wandb (optional, behind a flag).
- Never fake, guess or hard-code results. Every number in the report must come from a logged run. If something fails or a result is negative, report it honestly.
- Reproducibility: global seeding, every experiment driven by a YAML config, configs and git hash saved with each run, deterministic data order.
- Tests first for every core component (see the acceptance tests in each phase).
- Type hints, docstrings, small focused modules. No notebooks for core logic (notebooks only for exploration).
- Assume one GPU (~24 GB), but keep all sizes configurable so it also runs on CPU with tiny configs for tests. Use uv for environment management.
- When you have to make a design decision, pick the standard research choice, write it down in DECISIONS.md with one line of reasoning, and carry on.

## Repo layout

```
tinygpt-lab/
  PLAN.md  DECISIONS.md  README.md  pyproject.toml
  configs/            # yaml: tokenizer, model sizes, train, sae, sweeps
  src/tinygpt/
    tokenizer/bpe.py
    data/prepare.py  data/loader.py
    model/gpt.py  model/layers.py  model/generate.py
    train/trainer.py  train/optim.py  train/metrics.py
    scaling/run_sweep.py  scaling/fit.py
    sae/collect.py  sae/sae.py  sae/train_sae.py  sae/eval_sae.py
    interp/features.py  interp/dashboard.py  interp/interventions.py
    utils/seed.py  utils/logging.py  utils/checkpoint.py
  scripts/            # thin CLI entry points
  tests/
  runs/               # gitignored outputs
  report/             # paper.md, figures/
```

## Phase 0 — Setup

- Initialise repo, uv env, pyproject.toml, pytest, ruff, pre-commit, .gitignore (ignore runs/, data, checkpoints).
- Utilities: seeding, YAML config loading with dataclass validation, run directory creation (config + git hash + timestamp), JSONL metric logger.
- Accept: pytest passes; scripts/hello_run.py creates a run dir with saved config.

## Phase 1 — BPE tokenizer from scratch

- Byte-level BPE: train on a text corpus, encode, decode, save/load. Vocab size configurable (default 8192). Handle special tokens (<|endoftext|>).
- Make training efficient (pair-count updates, not a full recount every merge).
- Accept: round-trip decode(encode(x)) == x on random unicode text; merge determinism test; report compression ratio (bytes per token) on held-out text.

## Phase 2 — Data pipeline

- Dataset: TinyStories by default (config option for a FineWeb-Edu sample).
- Tokenize into uint16 memmap shards (train/val split). The loader samples random contiguous windows of length block_size.
- Accept: loader test (shapes, dtype, targets are inputs shifted by one); report total token count.

## Phase 3 — The model

Decoder-only transformer, modern choices:

- Pre-norm with RMSNorm, RoPE positional encoding, SwiGLU MLP, causal multi-head attention. Implement attention manually and through F.scaled_dot_product_attention, switchable by flag; tests confirm they match.
- Tied input/output embeddings, scaled init for residual projections (std / sqrt(2·n_layers)), no bias terms.
- Parameter counting (with and without embeddings) and a FLOPs-per-token estimate.
- Generation: KV cache, temperature, top-k, top-p.
- Accept:
  - Causality test: changing token t+1 never changes logits at positions ≤ t.
  - Manual and SDPA attention agree to 1e-5.
  - KV-cache generation matches no-cache generation exactly (greedy).
  - The model overfits a single batch to near-zero loss.

## Phase 4 — Training

- AdamW with decoupled weight decay (no decay on norms or embeddings), linear warmup + cosine decay, gradient clipping, gradient accumulation, bf16 autocast, optional torch.compile.
- Checkpointing with full resume (model, optimiser, scheduler, RNG states, data position). A test confirms resume is bit-identical over a few steps.
- Log: train/val loss, LR, grad norm, tokens/sec, MFU estimate, and samples generated every N steps.
- Accept: train a ~10M-param model on TinyStories to a sensible val loss, with coherent samples. Plot loss curves.

## Phase 5 — Scaling-law study

- First do an LR sweep at the smallest size (5–7 LRs, log-spaced) and pick the best. Note in DECISIONS.md how the LR is transferred to larger sizes (simple rule or µP-style scaling; implement µP only if time allows).
- Train 4–5 sizes (~1M, 3M, 10M, 25M, 50M non-embedding params) at a fixed tokens-per-parameter ratio (~20, Chinchilla-style), with the same data order.
- Fit L(N) = E + A · N^(−α) with scipy.optimize.curve_fit, plus bootstrap confidence intervals over the fit. Also plot loss against compute (FLOPs).
- If budget allows: an isoFLOP mini-study at 2–3 compute budgets to estimate the compute-optimal N.
- Accept: report/figures/scaling_law.png and a table with the fitted E, A, α and their CIs. Discuss how good the fit is and where it breaks down.

## Phase 6 — Collect activations

- From the largest model, record residual-stream activations at a middle layer (configurable) over ~50–100M tokens, stored as fp16 shards, together with the token ids they came from.
- Accept: shape/dtype checks; activation norm statistics logged.

## Phase 7 — Sparse autoencoder

- Implement a TopK SAE (and an L1 variant for comparison): expansion factor 8–16×, decoder columns constrained to unit norm, pre-encoder bias, input normalisation.
- Handle dead latents (an auxiliary loss or resampling; document the choice).
- Metrics: reconstruction MSE, fraction of variance explained, L0 sparsity, % dead latents, and most importantly downstream CE loss increase when the SAE reconstruction is spliced back into the model (compared with zero-ablation).
- Sweep k (or the L1 coefficient) and plot the sparsity vs. fidelity Pareto frontier.
- Accept: unit tests for the SAE (decoder norm constraint holds, TopK gives exactly k active latents); Pareto plot; chosen SAE checkpoint justified.

## Phase 8 — Interpreting features

- For each latent: top-activating contexts, activation histogram, the logit effect of the decoder direction (top boosted/suppressed tokens).
- Build a static HTML feature dashboard (one page, browsable, highlighted tokens).
- Hand-label ~50 randomly sampled features (I'll help review), assigning each a confidence score: interpretable / partly / not.
- Causal tests: for a few clear features (e.g. a "dialogue" feature or a "past tense" feature), (a) ablate the feature and measure the change in model outputs, (b) steer by adding the decoder direction and show generations shifting. Report effect sizes, not just examples.
- Accept: dashboard HTML, labelled-feature table, intervention results with quantitative metrics.

## Phase 9 — Original extension (pick one; I'll confirm)

Propose 2–3 options with a one-paragraph plan each and wait for my choice. Starting ideas:

- Feature universality across scale: train SAEs on 2–3 model sizes and match features across them (decoder cosine similarity / activation correlation). Which features appear only at larger scale?
- Feature emergence during training: train SAEs on checkpoints along one run. When do features appear, and does their emergence line up with drops in the loss curve?
- Scaling of SAE quality: how do fidelity, dead-latent fraction and interpretability change with model size at a fixed expansion factor?

## Phase 10 — Write-up

- report/paper.md in paper format: Abstract, Introduction, Related Work (cite the key papers: Vaswani 2017, Kaplan 2020, Hoffmann 2022, Bricken et al. 2023, Templeton et al. 2024, Gao et al. 2024 TopK SAEs), Method, Experiments, Results, Negative Results & Limitations, Conclusion.
- Every figure generated by a script in scripts/make_figures.py from logged runs.
- README.md: one-paragraph summary, headline figures, exact commands to reproduce each result, compute used (GPU-hours).
- Accept: a fresh clone + the README commands reproduces the figures (tested with the tiny CPU configs).

## How to work with me

- Before Phase 1, give me a short plan of estimated GPU-hours per phase so I can budget compute.
- Keep runs small until a component is verified; only then scale up.
- If a run diverges or the results look wrong, debug systematically (check LR, init, data, masking) and explain what you found before rerunning.
- Explain non-obvious design choices briefly as you go. I want to understand every part of this codebase well enough to defend it in an interview.

Start with Phase 0 now.
