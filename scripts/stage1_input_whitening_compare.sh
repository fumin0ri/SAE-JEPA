#!/usr/bin/env bash
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
RUN_ROOT="${RUN_ROOT:-runs/stage1-input-whitening-100k}"
STEPS="${STEPS:-100000}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
SKIP_LEADING_POSITIONS="${SKIP_LEADING_POSITIONS:-1}"
mkdir -p "$RUN_ROOT"
git rev-parse HEAD > "$RUN_ROOT/code-commit.txt"
python -m pip freeze > "$RUN_ROOT/python-environment.txt"
for mode in scalar zca; do
  whitening_path=""
  if [[ "$mode" == zca ]]; then whitening_path="$INPUT_WHITENING"; fi
  sj-masked train --config configs/sigreg_only.yaml \
    --set "name=sigreg-only-input-$mode" \
    --set "data.activation_manifest=$ACTIVATION_MANIFEST" \
    --set "data.normalization_path=$NORMALIZATION" \
    --set "data.input_whitening_path='$whitening_path'" \
    --set "data.skip_leading_positions=$SKIP_LEADING_POSITIONS" \
    --set "optim.steps=$STEPS" --set "train.seed=$SEED" \
    --set "train.device=$DEVICE" \
    --set "train.output_dir=$RUN_ROOT/$mode/seed-$SEED" \
    --set eval.input_diagnostics=true
done
sj-masked report --run-root "$RUN_ROOT"
