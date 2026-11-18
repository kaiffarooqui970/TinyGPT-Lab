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

<!-- RESULTS: filled from report/results/*.json once the runs finish. -->

## 5. Negative results and limitations

<!-- Filled from the logged results. -->

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
