#!/usr/bin/env bash
# Phase 4-5 compute queue, smallest first so problems surface early. Resumable: completed
# runs are skipped. Wrapped in caffeinate so macOS does not throttle or sleep.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
run() { caffeinate -ims uv run python scripts/run_sweep.py "$@"; }
run --config configs/sweeps/scaling.yaml --only s1 s2 s3
run --config configs/sweeps/scaling_lrcheck.yaml --only s3
run --config configs/sweeps/scaling.yaml --only s4
run --config configs/sweeps/scaling.yaml --only s5
echo QUEUE_DONE
