#!/usr/bin/env bash
# Fit only the lambda=0 Stage1 control, then diagnose it and existing runs.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
: "${ACTIVATION_MANIFEST:?Set ACTIVATION_MANIFEST}"
: "${NORMALIZATION:?Set NORMALIZATION}"
CONTROL_ROOT="${CONTROL_ROOT:-runs/whitened-residual-control-100k}"
EXISTING_ROOT="${EXISTING_ROOT:-runs/whitened-residual-100k}"
DIAGNOSTIC_ROOT="${DIAGNOSTIC_ROOT:-runs/whitened-residual-paired-diagnostics}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  RUN_ROOT="$CONTROL_ROOT" WEIGHTS=0 STEPS=100000 STAGE2=0 SEED="$SEED" DEVICE="$DEVICE" \
    bash scripts/stage1_whitened_residual.sh
fi
for weight in 0 0p001 0p01; do
  root="$EXISTING_ROOT"
  if [[ "$weight" == "0" ]]; then root="$CONTROL_ROOT"; fi
  for precision in none bfloat16; do
    python -m sae_jepa.paired_whitening \
      --checkpoint "$root/stage1/lambda-$weight/seed-$SEED/checkpoints/latest.pt" \
      --activation-manifest "$ACTIVATION_MANIFEST" \
      --output "$DIAGNOSTIC_ROOT/lambda-$weight/seed-$SEED/$precision" \
      --batches 64 --amp-dtype "$precision" --device "$DEVICE"
  done
done
