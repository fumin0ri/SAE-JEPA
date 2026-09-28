#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
: "${ACTIVATION_MANIFEST:?Set ACTIVATION_MANIFEST}"
RUN_ROOT="${RUN_ROOT:-runs/stage1-masked}"
MASK_PROBS="${MASK_PROBS:-0.25 0.5}"
WEIGHTS="${WEIGHTS:-0.001 0.003}"
SEED="${SEED:-42}"
STEPS="${STEPS:-100000}"
DEVICE="${DEVICE:-cuda}"
SKIP_LEADING_POSITIONS="${SKIP_LEADING_POSITIONS:-1}"
mkdir -p "$RUN_ROOT"
git rev-parse HEAD > "$RUN_ROOT/code-commit.txt"
python -m pip freeze > "$RUN_ROOT/python-environment.txt"
# An existing AE normalization may be reused only with identical data/exclusions.
NORMALIZATION="${NORMALIZATION:-$RUN_ROOT/normalization.pt}"
if [[ ! -f "$NORMALIZATION" ]]; then
  sj-compute-normalization --activation-manifest "$ACTIVATION_MANIFEST" \
    --skip-leading-positions "$SKIP_LEADING_POSITIONS" --output "$NORMALIZATION"
fi
for probability in $MASK_PROBS; do
  for weight in $WEIGHTS; do
    name="mask-${probability//./p}-lambda-${weight//./p}"
    run_dir="$RUN_ROOT/$name/seed-$SEED"
    sj-masked train --config configs/masked_sigreg.yaml \
      --set "name=$name" --set "data.activation_manifest=$ACTIVATION_MANIFEST" \
      --set "data.normalization_path=$NORMALIZATION" \
      --set "data.skip_leading_positions=$SKIP_LEADING_POSITIONS" \
      --set "masking.probability=$probability" --set "sigreg.weight=$weight" \
      --set "optim.steps=$STEPS" --set "train.seed=$SEED" \
      --set "train.device=$DEVICE" --set "train.output_dir=$run_dir" "$@"
  done
done
sj-masked report --run-root "$RUN_ROOT"
