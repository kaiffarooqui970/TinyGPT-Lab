# Scaling Laws and Sparse Features in Small Language Models Trained from Scratch

*tinygpt-lab — draft. Every number in this document is read from logged runs
(`report/results/*.json`, produced by the scripts listed in the README).*

## Abstract

<!-- ABSTRACT: written last, from the results below. -->

## 1. Introduction

Two empirical regularities underpin much of what we know about language models. First, their
loss falls smoothly, as a power law, with parameters, data and compute (Kaplan et al., 2020;
Hoffmann et al., 2022). Second, their internal activations can be decomposed by sparse
dictionary learning into features that are often individually interpretable (Bricken et al.,
2023; Templeton et al., 2024; Gao et al., 2024). This project reproduces both regularities
end to end at small scale, with every component written from scratch in PyTorch: the
tokenizer, the transformer, the training loop, the sparse autoencoders, and the
interpretability tooling. It then asks a question that connects them: **do the features an
SAE finds in a larger model also exist in smaller models trained on the same data?**

Contributions:

1. A from-scratch, tested, reproducible pipeline (byte-level BPE → GPT with RoPE/SwiGLU/RMSNorm
   → AdamW training with exact resume → sweeps → SAEs → dashboards → causal interventions),
   runnable end to end on a laptop CPU at toy scale.
2. A five-point scaling study (0.43M–10.6M non-embedding parameters, 20 tokens per parameter)
   with a fitted L(N) = E + A·N^(−α), bootstrap confidence intervals and an out-of-sample
   extrapolation test.
3. TopK and L1 SAEs on the largest model, with the sparsity–fidelity frontier measured by the
   downstream cross-entropy (CE) increase, plus hand-labelled features and quantitative ablation
   and steering experiments.
4. A universality analysis across model sizes, with a seed-reproducibility ceiling and a
   token-identity baseline.

## 2. Related work

**Transformers.** We use a decoder-only transformer (Vaswani et al., 2017) with the now-standard
modifications: pre-normalisation with RMSNorm (Zhang & Sennrich, 2019), rotary position
embeddings (Su et al., 2021), SwiGLU feed-forward layers (Shazeer, 2020), and tied input and
output embeddings (Press & Wolf, 2017).

**Scaling laws.** Kaplan et al. (2020) found that test loss follows power laws in non-embedding
parameters N, dataset size D and compute C, and fitted the optimal learning rate as a function
of N. Hoffmann et al. (2022, "Chinchilla") revised the compute-optimal allocation to roughly 20
tokens per parameter and fitted L(N, D) = E + A/N^α + B/D^β. We train every model at that
20:1 ratio, so N and D grow together, and fit the one-variable form L(N) = E + A·N^(−α).

**Small-scale data.** TinyStories (Eldan & Li, 2023) is a synthetic corpus of simple stories.
Its narrow distribution lets models with only a few million parameters produce fluent text.
That makes it a natural testbed when compute is limited.

**Sparse dictionary learning.** Bricken et al. (2023) trained L1-penalised sparse autoencoders on
a one-layer transformer and found largely monosemantic features. Templeton et al. (2024)
scaled the approach to a production model and showed that steering along feature directions
changes behaviour. Gao et al. (2024) replaced the L1 penalty with a TopK activation, which sets
sparsity directly and removes activation shrinkage. They added an auxiliary loss (AuxK) that
revives dead latents and proposed downstream-loss evaluation. Our SAEs follow Gao et al.:
TopK with AuxK, unit-norm decoder columns, a pre-encoder bias, and input normalisation. An
L1 variant serves as the baseline.

**Universality.** Olah et al. (2020) conjectured that similar features recur across networks.
Bricken et al. (2023) compared SAE features between two transformers trained with different
seeds via activation correlation. We apply the same measure across model *sizes*. To show how
much mismatch comes from SAE training noise alone, we calibrate it against an SAE seed ceiling.

## 3. Method

### 3.1 Tokenizer and data

Byte-level BPE with a GPT-2-style pre-tokenization regex, implemented with incremental
pair-count updates (a pair → words index plus a lazily invalidated max-heap). Ties are broken by
the smallest pair, so training is deterministic. The vocabulary has 8,192 entries: 256 bytes,
7,935 merges, and `<|endoftext|>`. Merges are learned on the first 200k TinyStories training
stories. The full corpus is tokenized into uint16 memmap shards, with an EOT token after each
story. Training samples are random contiguous windows of 256 tokens drawn by one seeded
generator, so every model sees the same data order.

### 3.2 Model

Pre-norm decoder-only transformer: RMSNorm; RoPE (θ = 10⁴) applied to queries and keys; causal
multi-head attention, implemented both manually and through `scaled_dot_product_attention`
(tests confirm they agree to within 1e-5); and a SwiGLU MLP whose hidden size is 8/3·d, rounded
up to a multiple of 64. There are no biases. Embeddings are tied. Initialisation is N(0, 0.02),
with residual output projections scaled by 1/√(2L). N counts non-embedding parameters
(Kaplan et al.). Training FLOPs per token are 6N + 12·L·d·T + 6·d·V; the scaling plots use
C = 6ND.

### 3.3 Training

AdamW (β = 0.9, 0.95; weight decay 0.1 on matrices only), linear warmup over 5% of steps, then
cosine decay to 0.1× the peak; gradient clipping at 1.0; bf16 autocast; 8,192 tokens per
step. The learning rate is a pure function of the step. A checkpoint captures the model, the
optimiser, the loader's generator, and all RNG states, and a test verifies that resuming is
bit-identical. The final validation loss is the mean over 200 fixed batches of 32 × 256 tokens.

### 3.4 Scaling protocol

Five sizes, S1–S5, trained at D = 20·N with the same data order. The peak LR is tuned by a
7-point sweep at S1 and transferred to larger sizes with Kaplan et al.'s fit,
LR(N) ∝ 0.003239 − 0.0001395·ln N, anchored at the S1 optimum. We test the transfer with 0.5×
and 2× runs at S3. We fit L(N) = E + A·N^(−α) by least squares (`scipy.optimize.curve_fit`).
Confidence intervals come from a bootstrap that resamples each run's 200 per-batch validation
losses. As a predictive test, we fit the four smallest sizes and extrapolate to S5.

### 3.5 Sparse autoencoders

Activations x ∈ ℝ^d are taken from the residual stream after half the model's blocks.
Inputs are scaled so that E‖x‖² = d. With pre-encoder bias b_dec:
pre = W_enc(x − b_dec) + b_enc; z = ReLU(TopK(pre)) (or ReLU(pre) for L1);
x̂ = W_dec z + b_dec. Decoder columns are kept at unit norm: before each Adam step we project
out the gradient component parallel to each column, and after it we renormalise. For TopK, the
AuxK loss (coefficient 1/32) lets the top d/2 dead latents (those that have not fired for 2M
tokens) reconstruct the residual error. For L1, the penalty λ·Σz ramps up linearly over the
first 5% of training. Evaluation reports MSE, the fraction of variance explained (FVE), L0,
dead and dense fractions on held-out activations, and the **downstream CE** when x̂ replaces
x in the forward pass. We compare the latter with zero- and mean-ablation through
CE recovered = (CE_zero − CE_SAE)/(CE_zero − CE_clean).

### 3.6 Interpretation and causal tests

For each latent we compute its firing frequency, top-activating contexts, a log-activation
histogram, and its direct logit effect W_U(g ⊙ d_i). These feed a static HTML dashboard. We
sample 50 live latents uniformly at random (seed 0) and label each as *interpretable*,
*partly*, or *not*. For selected features we (a) **ablate**: subtract z_i·d_i at the SAE layer
and measure, on the positions where the feature fires, the KL divergence and the change in
probability of feature-relevant "watch" tokens, compared with 10 random features of similar
frequency; and (b) **steer**: add α·d_i at every position during sampling and measure the
watch-token rate, the feature's own activation, and fluency (NLL under the unsteered model).
We report bootstrap CIs and Cohen's d.

### 3.7 Universality across scale

SAEs with an identical recipe (16× expansion, k = 32, 2.1M training-token budget, same
held-out tokens) are trained at half depth of S3, S4 and S5, plus a second-seed SAE on S5.
For each S5 latent we compute the maximum Pearson correlation of its activation with any
latent of the other SAE, over 65,536 shared held-out tokens. The S5-vs-S5-seed distribution is
the ceiling. A latent is *scale-specific* if it is reproducible (seed correlation ≥ 0.7) but
unmatched in S3 and S4 (best correlation < 0.3). A token-identity baseline, the maximum
correlation with any single current-token indicator, separates lexical from contextual
features.

## 4. Experiments and results

### 4.1 Scaling law

| model | N (non-emb.) | D (tokens) | C = 6ND | peak LR | val loss |
|---|---|---|---|---|---|
| S1 | 426,624 | 8.5M | 2.2e13 | 4.80e-3 | 2.616 |
| S2 | 953,440 | 19.1M | 1.1e14 | 4.42e-3 | 2.221 |
| S3 | 1,771,200 | 35.4M | 3.8e14 | 4.13e-3 | 1.997 |
| S4 | 4,820,224 | 96.4M | 2.8e15 | 3.67e-3 | 1.750 |
| S5 | 10,621,824 | 212.4M | 1.4e16 | 3.30e-3 | 1.608 |

The fitted law is **L(N) = 1.282 + 376.9 · N^(−0.435)**. The 95% bootstrap CIs are
E ∈ [1.235, 1.324], A ∈ [274, 511] and α ∈ [0.409, 0.461] (Figure `scaling_law.png`). The fit
is tight: RMSE 0.0025 nats, with all residuals below 0.005 nats, which is comparable to the
standard error of each run's validation loss (≈ 0.0046–0.0051). **Extrapolation test:** fitting
S1–S4 only (α = 0.452, E = 1.315) predicts 1.619 for S5, against 1.608 measured. The 0.011-nat
miss is about 2× the eval noise, and the full fit is slightly flatter. In compute,
L(C) = 1.282 + B·C^(−0.218) (α_C 95% CI [0.206, 0.231]), which is consistent with α_N / 2,
as expected when D ∝ N.

The LR transfer holds up at S3: 0.5×, 1× and 2× the transferred LR reach 2.0135, 1.9974 and
2.0035, so the transferred value is the best of the three, and the optimum is flat
(±0.016 nats over a 4× LR range).

### 4.2 Sparse autoencoders on S5

All SAEs use 16× expansion (6,144 latents) on S5's residual stream after block 3. Each is
trained for 16.8M tokens (9.1 epochs over 1.84M activations) and evaluated on 262,144
held-out activations. CE is measured on validation tokens the SAE never saw. Clean CE is
1.589, and zero-ablation gives 9.011.

| SAE | L0 | FVE | dead | ΔCE (nats) | CE recovered |
|---|---|---|---|---|---|
| TopK k=8 | 8 | 0.906 | 21.4% | 0.115 | 98.46% |
| TopK k=16 | 16 | 0.931 | 0.6% | 0.066 | 99.12% |
| **TopK k=32** | 32 | 0.946 | 0.0% | 0.046 | 99.38% |
| TopK k=64 | 64 | 0.955 | 0.0% | 0.034 | 99.54% |
| TopK k=128 | 128 | 0.963 | 0.0% | 0.026 | 99.65% |
| L1 λ=0.1 | 781 | 0.976 | 0.0% | 0.016 | 99.79% |
| L1 λ=0.3 | 472 | 0.953 | 0.0% | 0.039 | 99.48% |
| L1 λ=1 | 124 | 0.890 | 0.0% | 0.128 | 98.28% |
| L1 λ=3 | 13 | 0.770 | 0.0% | 0.547 | 92.63% |
| L1 λ=10 | 1.0 | 0.167 | 0.0% | 5.143 | 30.71% |

**TopK dominates L1 at every sparsity level** (Figure `sae_pareto.png`). At a matched
L0 ≈ 13–16, TopK costs 0.066 nats against 0.547 for L1, and L1 latents at low λ are often
dense (34% fire on more than 10% of tokens at λ = 0.1). We pick **k = 32** as the working SAE:
it is the knee of the curve, has no dead latents, and loses only 0.046 nats. Going to k = 128
saves a further 0.02 nats at 4× the L0. At k = 8, AuxK does not prevent 21% of latents from
dying.

### 4.3 Interpreting features

<!-- PENDING: hand labels of 50 random latents (report/results/label_sheet.md) and the
ablation / steering runs (scripts/run_interventions.py) once features are chosen. -->

### 4.4 Universality across scale

SAE quality at the fixed recipe barely depends on model size. FVE is 0.957 / 0.948 / 0.945
and ΔCE is 0.047 / 0.047 / 0.046 nats for S3 / S4 / S5 (Figure `sae_quality_vs_size.png`),
and none of them has dead latents. A second-seed S5 SAE gives the same numbers (FVE 0.946,
ΔCE 0.046).

| S5 latents matched in… | median best corr. | ≥ 0.5 | ≥ 0.7 | ≥ 0.9 |
|---|---|---|---|---|
| S5, other SAE seed (ceiling) | 0.736 | 81.7% | 55.3% | 20.6% |
| S4 | 0.598 | 62.2% | 36.4% | 10.3% |
| S3 | 0.518 | 52.2% | 29.1% | 7.5% |

Feature overlap falls steadily with the size gap (Figure `universality_hist.png`). In the
other direction, smaller models' features are mostly found in S5: S3 latents have a median
best match of 0.655 in S5, and S4 latents 0.662. This fits the picture of a larger model that
keeps most smaller-model features and adds new ones. Of the 3,398 S5 latents that are
reproducible across seeds (correlation ≥ 0.7), **102 (3.0%) are scale-specific**, with no
counterpart above 0.3 in S3 or S4, and 1,609 are shared across all three sizes (≥ 0.7). The
two groups differ in kind. Scale-specific latents are **contextual rather than lexical**:
their median correlation with the best single-token indicator is 0.14, against 0.77 for
shared latents. They also fire more often (median frequency 0.78% vs 0.33%). Their top
contexts (`universality_examples.md`) include "the other [animals]" (latent 400, plural noun
after "the other"), the object of "ate" (latent 126), and the activity after "loved to"
(latent 169). These are syntactic or semantic roles that a token detector cannot express.

## 5. Negative results and limitations

- **Small scale range.** N spans 0.43M–10.6M (25×). A planned 25M point was dropped for the
  compute budget (laptop M4 for training; €2 of rented GPU for the SAEs), and 50M would need
  more tokens than TinyStories has at 20 tokens/param. The fitted E and α describe this range
  only, and the extrapolation test already shows the curve flattening slightly at S5.
- **Bootstrap CIs cover eval noise only.** Each size was trained once, so seed-to-seed
  variance is not included and the CIs are optimistic.
- **Constant batch size** (8,192 tokens) for all sizes. The larger models may be under-batched.
- **Small SAE training sets.** Disk limits cut the activation store to 2.1M tokens per model
  (1.84M for training), so each SAE sees its data about 9 times, against 50–100M fresh tokens
  in the plan. Fidelity is still high (99.4% CE recovered), but rarer features are probably
  missed.
- **Untuned SAE learning rate.** At k = 32, lr 1e-3 beats the default 4e-4 (ΔCE 0.037 vs
  0.046). We kept 4e-4 everywhere so that runs stay comparable.
- **Activation correlation is not identity.** A best-match correlation below 0.3 can also mean
  that a feature is split or merged differently, not that it is absent. The seed ceiling
  (55% of latents ≥ 0.7) shows that SAEs only partly reproduce themselves even at one size.

## 6. Conclusion

<!-- Written last. -->

## References

- Bricken, T. et al. (2023). *Towards Monosemanticity: Decomposing Language Models With
  Dictionary Learning.* Transformer Circuits Thread.
- Eldan, R. & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak
  Coherent English?* arXiv:2305.07759.
- Gao, L. et al. (2024). *Scaling and Evaluating Sparse Autoencoders.* arXiv:2406.04093.
- Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models.*
  arXiv:2203.15556.
- Kaplan, J. et al. (2020). *Scaling Laws for Neural Language Models.* arXiv:2001.08361.
- Olah, C. et al. (2020). *Zoom In: An Introduction to Circuits.* Distill.
- Press, O. & Wolf, L. (2017). *Using the Output Embedding to Improve Language Models.* EACL.
- Shazeer, N. (2020). *GLU Variants Improve Transformer.* arXiv:2002.05202.
- Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
  arXiv:2104.09864.
- Templeton, A. et al. (2024). *Scaling Monosemanticity: Extracting Interpretable Features from
  Claude 3 Sonnet.* Transformer Circuits Thread.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Zhang, B. & Sennrich, R. (2019). *Root Mean Square Layer Normalization.* NeurIPS.
