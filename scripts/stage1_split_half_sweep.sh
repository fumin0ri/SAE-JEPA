#!/usr/bin/env bash
# Dense reconstruction + SIGReg + split-half covariance: lambda x beta grid.
# Same settings as the plug-in gaussian-sweep (m=1024, k=256, 100k steps,
# decay 0.5, shared normalization) so runs pair up one-to-one with it.
# Finished runs (final eval present) are skipped, so the script can be rerun.
# Usage: ACTIVATION_MANIFEST=/path/manifest.json bash scripts/stage1_split_half_sweep.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${ACTIVATION_MANIFEST:?Set ACTIVATION_MANIFEST}"
RUN_ROOT="${RUN_ROOT:-runs/stage1}"
NORMALIZATION="${NORMALIZATION:-runs/stage1/rec-sigreg-cov/normalization.pt}"
LAMBDAS="${LAMBDAS:-0.1 0.03 0.01}"
BETAS="${BETAS:-100 300 1000}"
ESTIMATOR="${ESTIMATOR:-split_half}"
PROJECTIONS="${PROJECTIONS:-1024}"
SKETCH_DIM="${SKETCH_DIM:-256}"
STEPS="${STEPS:-100000}"
DECAY="${DECAY:-0.5}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
EXTRA_ARGS=(${EXTRA_ARGS:-})

[[ -f "$NORMALIZATION" ]] || { echo "missing normalization $NORMALIZATION" >&2; exit 1; }
mkdir -p "$RUN_ROOT"
final_eval="eval-validation-step-$(printf '%07d' "$STEPS").json"
steps_tag="$((STEPS / 1000))k"
decay_tag="decay$(python -c "print(round($DECAY * 100))")"

for lambda in $LAMBDAS; do
  for beta in $BETAS; do
    name="${ESTIMATOR//_/-}-sweep-m-$PROJECTIONS-lambda-$lambda-beta-$beta-$steps_tag-$decay_tag"
    run_dir="$RUN_ROOT/$name"
    if [[ -f "$run_dir/$final_eval" ]]; then
      echo "Skipping $name (finished)"
      continue
    fi
    echo "Training $name"
    mkdir -p "$run_dir"
    git rev-parse HEAD > "$run_dir/code-commit.txt" 2>/dev/null || true
    sj-train-dense --config configs/dense_sigreg_cov_ae.yaml \
      --set "data.activation_manifest=$ACTIVATION_MANIFEST" \
      --set "data.normalization_path=$NORMALIZATION" \
      --set "sigreg.weight=$lambda" \
      --set "sigreg.num_projections=$PROJECTIONS" \
      --set "covariance.weight=$beta" \
      --set "covariance.sketch_dim=$SKETCH_DIM" \
      --set "covariance.estimator=$ESTIMATOR" \
      --set "optim.steps=$STEPS" \
      --set "optim.decay_fraction=$DECAY" \
      --set "train.seed=$SEED" \
      --set "train.device=$DEVICE" \
      --set "train.output_dir=$run_dir" \
      "${EXTRA_ARGS[@]}"
  done
done
echo "Done. Final evaluations: $RUN_ROOT/*-sweep-*/$final_eval"
