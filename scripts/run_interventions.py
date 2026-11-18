"""Phase 8 causal tests: ablation and steering for selected SAE features, with controls
and bootstrap effect sizes.

Usage:
    uv run python scripts/run_interventions.py --config configs/interventions.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import torch

from tinygpt.data.loader import TokenLoader
from tinygpt.data.prepare import resolve, shard_paths
from tinygpt.interp.interventions import (
    ablation_effect,
    bootstrap_diff_ci,
    bootstrap_mean_ci,
    decoder_dir,
    steered_generations,
)
from tinygpt.model.gpt import load_gpt
from tinygpt.sae.collect import eval_min_start
from tinygpt.sae.train_sae import load_sae
from tinygpt.tokenizer.bpe import EOT, BPETokenizer
from tinygpt.train.metrics import pick_device
from tinygpt.utils.config import load_config
from tinygpt.utils.seed import seed_everything


@dataclass
class FeatureSpec:
    name: str
    feature: int
    watch: list[str]  # strings whose token-ids are tracked (e.g. ['"', ' said'])
    steer_multipliers: list[float] = field(default_factory=lambda: [4.0, 8.0])


@dataclass
class InterventionConfig:
    sae: str
    features: list[FeatureSpec]
    prompts: list[str]
    tokenizer: str = "data/tokenizer/tinystories_8k.json"
    data_dir: str = "data/tinystories"
    eval_batches: int = 40
    eval_batch_size: int = 32
    n_controls: int = 10
    samples_per_prompt: int = 10
    max_new_tokens: int = 64
    seed: int = 0
    out: str = "report"


def watch_ids(tok: BPETokenizer, words: list[str]) -> list[int]:
    ids = []
    for w in words:
        e = tok.encode(w)
        if len(e) != 1:
            raise ValueError(f"watch string {w!r} is not a single token: {e}")
        ids.append(e[0])
    return ids


@torch.no_grad()
def nll_under(model, seqs: list[list[int]], prompt_len: list[int], device) -> np.ndarray:
    """Mean per-token NLL of each generated continuation under the (unsteered) model."""
    out = []
    for s, pl in zip(seqs, prompt_len, strict=True):
        x = torch.tensor([s], device=device)
        logits, _, _ = model(x[:, :-1])
        lp = torch.log_softmax(logits.float(), -1)[0, pl - 1 :]
        tgt = x[0, pl:]
        out.append(float(-lp.gather(-1, tgt[:, None]).mean()))
    return np.array(out)


@torch.no_grad()
def feature_rate(model, sae, layer: int, seqs: list[list[int]], feature: int, device) -> np.ndarray:
    """Mean activation of `feature` over each sequence (re-encoded, no steering)."""
    out = []
    for s in seqs:
        r = model.residual(torch.tensor([s], device=device), layer)
        out.append(float(sae.encode(r)[0, :, feature].mean()))
    return np.array(out)


def measure(
    gens: list[list[int]],
    model,
    sae,
    layer: int,
    prompt_rep: list[list[int]],
    plen: list[int],
    wid: list[int],
    f: int,
    device,
) -> dict[str, np.ndarray]:
    """Per sample: watch-token rate and feature activation over the continuation, and NLL
    of the continuation under the unsteered model."""
    full = [p + g for p, g in zip(prompt_rep, gens, strict=True)]
    rate = np.array([np.isin(g, wid).mean() if len(g) else 0.0 for g in gens])
    return {
        "watch_rate": rate,
        "feature_act": feature_rate(model, sae, layer, full, f, device),
        "nll": nll_under(model, full, plen, device),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config, InterventionConfig)
    seed_everything(cfg.seed)
    device = pick_device()
    sae, ck = load_sae(cfg.sae, device)
    meta = ck["acts_meta"]
    layer = meta["layer"]
    model, _ = load_gpt(resolve(meta["checkpoint"]), device)
    tok = BPETokenizer.load(resolve(cfg.tokenizer))
    eos = tok.special_tokens[EOT]
    val = TokenLoader(
        shard_paths(cfg.data_dir, "val"),
        model.cfg.block_size,
        cfg.eval_batch_size,
        seed=0,
        min_start=eval_min_start(meta),
    )
    batches = list(val.fixed_batches(cfg.eval_batches, seed=999))
    stats = np.load(resolve(cfg.sae).parent / "feature_stats.npz")
    freq = stats["freq"]
    rng = np.random.default_rng(cfg.seed)
    results: dict[str, dict] = {}

    for spec in cfg.features:
        f = spec.feature
        if f < 0:
            # Auto mode (tiny reproduction): the |f|-th most frequent non-dense latent.
            order = np.argsort(-np.where(freq < 0.1, freq, -1.0))
            f = int(order[-spec.feature - 1])
        wid = watch_ids(tok, spec.watch)
        # ---- (a) ablation, on positions where the feature fires
        ab = ablation_effect(model, sae, f, layer, batches, device, wid)
        # Controls: random alive features with firing frequency within x2 of the target.
        cands = np.flatnonzero((freq > freq[f] / 2) & (freq < freq[f] * 2))
        cands = cands[cands != f]
        ctrl_ids = rng.choice(cands, size=min(cfg.n_controls, cands.size), replace=False)
        ctrl_watch, ctrl_kl = [], []
        for c in ctrl_ids:
            r = ablation_effect(model, sae, int(c), layer, batches, device, wid)
            ctrl_watch.append(
                r["delta_watch_prob"].mean() if r["delta_watch_prob"].size else np.nan
            )
            ctrl_kl.append(r["kl"].mean() if r["kl"].size else np.nan)
        ablation = {
            "n_positions": int(ab["kl"].size),
            "delta_loss_on_active": bootstrap_mean_ci(ab["delta_loss"]),
            "kl_on_active": bootstrap_mean_ci(ab["kl"]),
            "delta_watch_prob": bootstrap_mean_ci(ab["delta_watch_prob"]),
            "controls_delta_watch_prob": [float(x) for x in ctrl_watch],
            "controls_kl": [float(x) for x in ctrl_kl],
            "control_ids": ctrl_ids.tolist(),
        }

        # ---- (b) steering
        prompts = [tok.encode(p) for p in cfg.prompts]
        base = steered_generations(
            model,
            prompts,
            layer,
            None,
            cfg.samples_per_prompt,
            cfg.max_new_tokens,
            cfg.seed,
            device,
            eos,
        )
        plen = [len(p) for p in prompts for _ in range(cfg.samples_per_prompt)]

        prompt_rep = [p for p in prompts for _ in range(cfg.samples_per_prompt)]
        ctx = (model, sae, layer, prompt_rep, plen, wid, f, device)

        m0 = measure(base, *ctx)
        typical = float(stats["top_vals"][0, f])  # the feature's max activation (stats set)
        steer = {}
        examples = {"baseline": tok.decode(prompts[0] + base[0])}
        for mult in spec.steer_multipliers:
            vec = decoder_dir(sae, f) * typical * mult / 4.0
            gens = steered_generations(
                model,
                prompts,
                layer,
                vec,
                cfg.samples_per_prompt,
                cfg.max_new_tokens,
                cfg.seed,
                device,
                eos,
            )
            m1 = measure(gens, *ctx)
            steer[str(mult)] = {
                k: bootstrap_diff_ci(m1[k], m0[k])
                | {"steered_mean": float(m1[k].mean()), "baseline_mean": float(m0[k].mean())}
                for k in m1
            }
            examples[f"x{mult}"] = tok.decode(prompts[0] + gens[0])
        results[spec.name] = {
            "feature": f,
            "freq": float(freq[f]),
            "watch": spec.watch,
            "ablation": ablation,
            "steering": steer,
            "steer_scale_note": "vector = d_i * max_act * mult / 4",
            "examples": examples,
        }
        print(
            json.dumps(
                {
                    spec.name: {
                        "ablation_watch": ablation["delta_watch_prob"],
                        "kl": ablation["kl_on_active"],
                        "steer": {k: v["watch_rate"] for k, v in steer.items()},
                    }
                },
                indent=1,
            ),
            flush=True,
        )

    out = resolve(cfg.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / "interventions.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
