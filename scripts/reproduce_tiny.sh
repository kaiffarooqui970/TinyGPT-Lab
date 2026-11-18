#!/usr/bin/env bash
# End-to-end CPU reproduction of every phase at toy scale (fresh clone -> report_tiny/).
# Downloads only the TinyStories validation split (streamed). Takes ~10-20 min on a laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
py() { uv run python "$@"; }
T=configs/tiny
py scripts/train_tokenizer.py --config $T/tokenizer.yaml                     # Phase 1
py scripts/prepare_data.py --config $T/data.yaml                             # Phase 2
py scripts/run_sweep.py --config $T/lr_sweep.yaml                            # Phase 5 (LR)
py scripts/run_sweep.py --config $T/scaling.yaml                             # Phases 4-5
py scripts/collect_acts.py --config $T/collect_t3.yaml                       # Phase 6
py scripts/collect_acts.py --config $T/collect_t2.yaml
py scripts/train_sae.py --sweep $T/sae_sweep.yaml                            # Phase 7
py scripts/build_dashboard.py --sae "runs/sae/tiny_sae/*_topk_k8/sae.pt" \
   --tokens 200000 --n-label 20 --n-extra 30 --out report_tiny \
   --tokenizer data/tiny/tokenizer.json                                      # Phase 8
py scripts/run_interventions.py --config $T/interventions.yaml
py scripts/train_sae.py --sweep $T/sae_universality.yaml                     # Phase 9
py scripts/run_universality.py --out report_tiny --tokenizer data/tiny/tokenizer.json \
   --n-tokens 16384 --saes "s4=runs/sae/tiny_univ/*_t2/sae.pt" \
   "s5=runs/sae/tiny_univ/*_t3/sae.pt" "s5b=runs/sae/tiny_univ/*_t3_seed1/sae.pt"
py scripts/make_figures.py --config $T/figures.yaml                          # Phase 10
echo "TINY_REPRO_DONE: figures in report_tiny/figures"
