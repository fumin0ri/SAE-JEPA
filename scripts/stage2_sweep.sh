#!/usr/bin/env bash
# Stage-2 pilot over existing stage-1 checkpoints. No stage-1 retraining.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
: "${STAGE1_ROOT:?Set STAGE1_ROOT to the directory holding the intended stage-1 checkpoints}"
WEIGHTS="${WEIGHTS:-0 0.0003 0.001}"
STAGE1_SEED="${STAGE1_SEED:-42}"
RUN_ROOT="${RUN_ROOT:-runs/stage2-pilot}"
checkpoints=()
for weight in $WEIGHTS; do
  checkpoints+=("$STAGE1_ROOT/lambda-${weight//./p}/seed-$STAGE1_SEED/checkpoints/latest.pt")
done
args=(--checkpoints "${checkpoints[@]}" --output "$RUN_ROOT"
  --config configs/stage2_topk.yaml --device "${DEVICE:-cuda}"
  --set "seed=${SEED:-42}" --set "steps=${STEPS:-10000}"
  --set "dictionary_size=${DICTIONARY_SIZE:-16384}" --set "k=${K:-64}")
if [[ -n "${ACTIVATION_MANIFEST:-}" ]]; then args+=(--activation-manifest "$ACTIVATION_MANIFEST"); fi
if [[ "${RESUME:-0}" == "1" ]]; then args+=(--resume); fi
sj-stage2 sweep "${args[@]}" "$@"
git rev-parse HEAD > "$RUN_ROOT/code-commit.txt" 2>/dev/null || true
python -m pip freeze > "$RUN_ROOT/python-environment.txt" 2>/dev/null || true
echo "Results: $RUN_ROOT/report/validation.md"
