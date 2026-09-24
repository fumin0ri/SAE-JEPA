#!/usr/bin/env bash
# CPU smoke test of the full stage-1 pipeline on synthetic activations.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SMOKE_DIR="${SMOKE_DIR:-runs/smoke}"

sj-make-synthetic --output-dir "$SMOKE_DIR/data" --d-in 64 --train-shards 8 --validation-shards 4
ACTIVATION_MANIFEST="$SMOKE_DIR/data/manifest.json" \
RUN_ROOT="$SMOKE_DIR/runs" \
WEIGHTS="0 0.1" STEPS=60 DEVICE=cpu \
EXTRA_ARGS="--set model.d_hidden=64 --set model.d_latent=64 --set optim.batch_size=128 \
  --set optim.warmup_steps=10 --set optim.lr=1e-3 --set sigreg.num_projections=32 \
  --set sigreg.validation_projections=32 --set train.log_every=20 --set train.validation_every=30 \
  --set train.validation_batches=4 --set train.checkpoint_every=30 --set eval.batch_size=128 \
  --set eval.batches=8 --set eval.diagnostic_projections=32" \
  bash scripts/stage1_sweep.sh
cat "$SMOKE_DIR/runs/report/validation.md"
