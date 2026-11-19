#!/usr/bin/env bash
# Phases 6, 7, 9 (+ Phase 8 dashboard) on a rented CUDA GPU. Expects the repo, the val split,
# the tokenizer and the S3/S4/S5 final checkpoints to be present (see scripts/pod_sync.sh).
# Writes PIPELINE_DONE / PIPELINE_FAILED as the last line of runs/logs/pod_pipeline.log.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
mkdir -p runs/logs
LOG=runs/logs/pod_pipeline.log
trap 'echo "PIPELINE_FAILED at line $LINENO" | tee -a $LOG' ERR
py() { echo ">>> $*" | tee -a $LOG; python "$@" 2>&1 | tee -a $LOG; }

py scripts/collect_acts.py --config configs/collect/s5_l3.yaml           # Phase 6
py scripts/collect_acts.py --config configs/collect/s4_l3.yaml
py scripts/collect_acts.py --config configs/collect/s3_l2.yaml
py scripts/train_sae.py --sweep configs/sae/sweep_s5.yaml                 # Phase 7
py scripts/train_sae.py --sweep configs/sae/universality.yaml             # Phase 9 SAEs
py scripts/run_universality.py --out report \
   --saes "s3=runs/sae/universality/*_s3/sae.pt" "s4=runs/sae/universality/*_s4/sae.pt" \
   "s5=runs/sae/universality/*_s5/sae.pt" "s5b=runs/sae/universality/*_s5_seed1/sae.pt"
py scripts/build_dashboard.py --sae "runs/sae/sweep_s5/*_topk_k32/sae.pt" --out report  # Phase 8
echo PIPELINE_DONE | tee -a $LOG
