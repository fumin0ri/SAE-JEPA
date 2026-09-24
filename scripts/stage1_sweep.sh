#!/usr/bin/env bash
# Stage 1 pilot: Dense-AE (lambda=0) and Dense-SIGReg-AE (lambda>0) with identical
# data order, initialization, and train-only normalization statistics.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${ACTIVATION_MANIFEST:?Set ACTIVATION_MANIFEST to a JEPA-SAE sr-extract-pile manifest}"

RUN_ROOT="${RUN_ROOT:-runs/stage1-pilot}"
WEIGHTS="${WEIGHTS:-0 0.01 0.1 1}"
SEED="${SEED:-42}"
STEPS="${STEPS:-10000}"
DEVICE="${DEVICE:-cuda}"
EVAL_SPLITS="${EVAL_SPLITS:-validation test}"
EXTRA_ARGS=(${EXTRA_ARGS:-})

mkdir -p "$RUN_ROOT"
git rev-parse HEAD > "$RUN_ROOT/code-commit.txt" 2>/dev/null || true
python -m pip freeze > "$RUN_ROOT/python-environment.txt" 2>/dev/null || true

NORMALIZATION="$RUN_ROOT/normalization.pt"
if [[ ! -f "$NORMALIZATION" ]]; then
  echo "Computing train-only normalization statistics"
  sj-compute-normalization --activation-manifest "$ACTIVATION_MANIFEST" --output "$NORMALIZATION"
fi

for weight in $WEIGHTS; do
  name="lambda-${weight//./p}"
  if [[ "$weight" == "0" ]]; then config=configs/dense_ae.yaml; else config=configs/dense_sigreg_ae.yaml; fi
  run_dir="$RUN_ROOT/$name/seed-$SEED"
  echo "Training $name (seed $SEED)"
  sj-train-dense --config "$config" \
    --set "name=$name" \
    --set "data.activation_manifest=$ACTIVATION_MANIFEST" \
    --set "data.normalization_path=$NORMALIZATION" \
    --set "sigreg.weight=$weight" \
    --set "optim.steps=$STEPS" \
    --set "train.seed=$SEED" \
    --set "train.device=$DEVICE" \
    --set "train.output_dir=$run_dir" \
    "${EXTRA_ARGS[@]}"
  for split in $EVAL_SPLITS; do
    sj-evaluate-dense --checkpoint "$run_dir/checkpoints/latest.pt" --split "$split" --device "$DEVICE"
  done
done

sj-report-dense --run-root "$RUN_ROOT"
echo "Report written to $RUN_ROOT/report"
