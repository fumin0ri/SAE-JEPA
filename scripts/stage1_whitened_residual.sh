#!/usr/bin/env bash
# Whitened residual AE (y = z + G(z) on the PCA baseline's z) -> the same
# 64k/100k Top-K SAE as scripts/baseline_sae_100k.sh, optionally with probes.
# Run from any directory. RESUME=1 continues interrupted stage-1/stage-2 runs.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
: "${ACTIVATION_MANIFEST:?Set ACTIVATION_MANIFEST}"
: "${NORMALIZATION:?Set NORMALIZATION to the normalization the PCA baseline was built with}"
WHITENING="${WHITENING:-runs/baseline-raw-pca-64k-100k/frontends/pca.pt}"
for input_file in "$NORMALIZATION" "$WHITENING"; do
  if [[ ! -f "$input_file" ]]; then
    echo "Missing input: $input_file" >&2
    exit 1
  fi
done
RUN_ROOT="${RUN_ROOT:-runs/whitened-residual-100k}"
WEIGHTS="${WEIGHTS:-0.001 0.01}"
STEPS="${STEPS:-100000}"
DECAY_FRACTION="${DECAY_FRACTION:-0.5}"
SEED="${SEED:-42}"
SAE_SEEDS="${SAE_SEEDS:-42}"
DEVICE="${DEVICE:-cuda}"
SKIP_LEADING_POSITIONS="${SKIP_LEADING_POSITIONS:-0}"
STAGE2="${STAGE2:-1}"
EXTRA_ARGS=(${EXTRA_ARGS:-})

mkdir -p "$RUN_ROOT"
git rev-parse HEAD > "$RUN_ROOT/code-commit.txt"
python -m pip freeze > "$RUN_ROOT/python-environment.txt"

checkpoints=()
for weight in $WEIGHTS; do
  run_dir="$RUN_ROOT/stage1/lambda-${weight//./p}/seed-$SEED"
  echo "Stage 1: lambda=$weight (seed $SEED)"
  sj-train-dense --config configs/whitened_residual_ae.yaml \
    --set "name=whitened-residual-lambda-${weight//./p}" \
    --set "data.activation_manifest=$ACTIVATION_MANIFEST" \
    --set "data.normalization_path=$NORMALIZATION" \
    --set "data.input_whitening_path=$WHITENING" \
    --set "data.skip_leading_positions=$SKIP_LEADING_POSITIONS" \
    --set "sigreg.weight=$weight" \
    --set "optim.steps=$STEPS" --set "optim.decay_fraction=$DECAY_FRACTION" \
    --set "train.seed=$SEED" --set "train.device=$DEVICE" \
    --set "train.output_dir=$run_dir" \
    "${EXTRA_ARGS[@]}"
  checkpoints+=("$run_dir/checkpoints/latest.pt")
done
sj-report-dense --run-root "$RUN_ROOT/stage1"

if [[ "$STAGE2" != "1" ]]; then
  exit 0
fi
# Same SAE settings as the Raw/PCA baselines (reconstruction in activation space).
for sae_seed in $SAE_SEEDS; do
  output="$RUN_ROOT/stage2/seed-$sae_seed"
  args=(--checkpoints "${checkpoints[@]}"
    --config configs/stage2_topk.yaml
    --activation-manifest "$ACTIVATION_MANIFEST"
    --output "$output" --device "$DEVICE"
    --set steps=100000 --set dictionary_size=65536 --set k=64
    --set batch_size=512 --set "seed=$sae_seed"
    --set reconstruction_space=activation)
  if [[ "${RESUME:-0}" == "1" && -f "$output/comparison.json" ]]; then
    args+=(--resume)
  fi
  sj-stage2 sweep "${args[@]}"
  if [[ -n "${PROBE_TASKS:-}" && -n "${PROBE_ACTIVATIONS:-}" ]]; then
    sj-probe evaluate --tasks "$PROBE_TASKS" --activations "$PROBE_ACTIVATIONS" \
      --checkpoints "$output"/model-*/checkpoints/latest.pt \
      --output "$output/probe" --device "$DEVICE"
  fi
done
