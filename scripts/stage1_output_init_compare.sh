#!/usr/bin/env bash
# SIGReg-only on ZCA input: default initialization vs output layer calibrated to
# unit variance at step 0.  Both runs log output rank/PR at every validation.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
: "${ACTIVATION_MANIFEST:?Set ACTIVATION_MANIFEST}"
: "${NORMALIZATION:?Set existing train-only scalar NORMALIZATION}"
: "${INPUT_WHITENING:?Set train-fit INPUT_WHITENING .pt file}"
for input_file in "$NORMALIZATION" "$INPUT_WHITENING"; do
  if [[ ! -f "$input_file" ]]; then
    echo "Missing input statistics: $input_file" >&2
    exit 1
  fi
done
RUN_ROOT="${RUN_ROOT:-runs/stage1-output-init-100k}"
STEPS="${STEPS:-100000}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
DECAY_FRACTION="${DECAY_FRACTION:-0.5}"
INIT_VARIANCE="${INIT_VARIANCE:-1.0}"
SKIP_LEADING_POSITIONS="${SKIP_LEADING_POSITIONS:-1}"
mkdir -p "$RUN_ROOT"
git rev-parse HEAD > "$RUN_ROOT/code-commit.txt"
python -m pip freeze > "$RUN_ROOT/python-environment.txt"
for mode in default calibrated; do
  init_variance=0
  if [[ "$mode" == calibrated ]]; then init_variance="$INIT_VARIANCE"; fi
  sj-masked train --config configs/sigreg_only.yaml \
    --set "name=sigreg-only-zca-init-$mode" \
    --set "data.activation_manifest=$ACTIVATION_MANIFEST" \
    --set "data.normalization_path=$NORMALIZATION" \
    --set "data.input_whitening_path=$INPUT_WHITENING" \
    --set "data.skip_leading_positions=$SKIP_LEADING_POSITIONS" \
    --set "model.init_output_variance=$init_variance" \
    --set "optim.steps=$STEPS" --set "optim.decay_fraction=$DECAY_FRACTION" \
    --set "train.seed=$SEED" --set "train.device=$DEVICE" \
    --set "train.output_dir=$RUN_ROOT/$mode/seed-$SEED" \
    --set eval.input_diagnostics=true
done
sj-masked report --run-root "$RUN_ROOT"
