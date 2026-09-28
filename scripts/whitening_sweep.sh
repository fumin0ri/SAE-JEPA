#!/usr/bin/env bash
# PCA whitening control: raw / PCA / dense front-ends x latent / original loss,
# all with the same stage-2 budget, seed and SAE initialization.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
: "${STAGE1_ROOT:?Set STAGE1_ROOT to the directory holding the intended stage-1 checkpoints}"
WEIGHTS="${WEIGHTS:-0 0.001}"
STAGE1_SEED="${STAGE1_SEED:-42}"
RUN_ROOT="${RUN_ROOT:-runs/stage2-whitening}"
# Front-end files live outside RUN_ROOT (sj-stage2 sweep requires an empty output).
FRONTEND_DIR="${FRONTEND_DIR:-${RUN_ROOT}-frontends}"
LOSS_SPACES="${LOSS_SPACES:-latent original}"
PCA_POSITIONS="${PCA_POSITIONS:-2097152}"
PCA_RELATIVE_EPSILON="${PCA_RELATIVE_EPSILON:-1e-3}"
DEVICE="${DEVICE:-cuda}"
manifest=()
if [[ -n "${ACTIVATION_MANIFEST:-}" ]]; then manifest=(--activation-manifest "$ACTIVATION_MANIFEST"); fi

dense=()
for weight in $WEIGHTS; do
  dense+=("$STAGE1_ROOT/lambda-${weight//./p}/seed-$STAGE1_SEED/checkpoints/latest.pt")
done
# Raw and PCA copy data policy and normalization from the first dense checkpoint,
# so every candidate sees identical inputs (checked again by sj-stage2 preflight).
like="${dense[0]}"
mkdir -p "$FRONTEND_DIR"
if [[ ! -f "$FRONTEND_DIR/raw.pt" ]]; then
  sj-make-raw-frontend --like "$like" "${manifest[@]}" --output "$FRONTEND_DIR/raw.pt"
fi
if [[ ! -f "$FRONTEND_DIR/pca.pt" ]]; then
  sj-fit-pca --like "$like" "${manifest[@]}" --output "$FRONTEND_DIR/pca.pt" \
    --maximum-positions "$PCA_POSITIONS" --relative-epsilon "$PCA_RELATIVE_EPSILON" --device "$DEVICE"
fi

args=(--checkpoints "$FRONTEND_DIR/raw.pt" "$FRONTEND_DIR/pca.pt" "${dense[@]}"
  --loss-spaces $LOSS_SPACES --output "$RUN_ROOT"
  --config configs/stage2_topk.yaml --device "$DEVICE"
  --set "seed=${SEED:-42}" --set "steps=${STEPS:-10000}"
  --set "dictionary_size=${DICTIONARY_SIZE:-16384}" --set "k=${K:-64}")
args+=("${manifest[@]}")
if [[ "${RESUME:-0}" == "1" ]]; then args+=(--resume); fi
sj-stage2 sweep "${args[@]}" "$@"
git rev-parse HEAD > "$RUN_ROOT/code-commit.txt" 2>/dev/null || true
python -m pip freeze > "$RUN_ROOT/python-environment.txt" 2>/dev/null || true
echo "PCA diagnostics: $FRONTEND_DIR/pca.json"
echo "Results: $RUN_ROOT/report/validation.md"
