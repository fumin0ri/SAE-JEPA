#!/usr/bin/env bash
# Raw / paper-style PCA whitening -> Top-K SAE. Run from any directory.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
: "${REFERENCE:?Set REFERENCE to a full-dimensional Stage1 checkpoint}"
: "${ACTIVATION_MANIFEST:?Set ACTIVATION_MANIFEST}"
RUN_ROOT="${RUN_ROOT:-runs/baseline-raw-pca-64k-100k}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42}"
BASELINES="$RUN_ROOT/frontends"

if [[ "${RESUME:-0}" != "1" ]]; then
  sj-stage2 prepare-baselines \
    --reference-checkpoint "$REFERENCE" \
    --activation-manifest "$ACTIVATION_MANIFEST" \
    --output "$BASELINES" --kinds raw pca \
    --maximum-positions 262144 --sample-seed 1729 \
    --epsilon 1e-4 --chunk-size 2048 --device "$DEVICE"
fi

for seed in $SEEDS; do
  args=(--checkpoints "$BASELINES/raw.pt" "$BASELINES/pca.pt"
    --config configs/stage2_topk.yaml
    --activation-manifest "$ACTIVATION_MANIFEST"
    --output "$RUN_ROOT/seed-$seed" --device "$DEVICE"
    --set steps=100000 --set dictionary_size=65536 --set k=64
    --set batch_size=512 --set "seed=$seed"
    --set reconstruction_space=activation)
  if [[ "${RESUME:-0}" == "1" && -f "$RUN_ROOT/seed-$seed/comparison.json" ]]; then
    args+=(--resume)
  fi
  sj-stage2 sweep "${args[@]}"
done
