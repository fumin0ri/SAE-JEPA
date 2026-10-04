#!/usr/bin/env bash
# Downstream comparison of arbitrary stage-1 runs: Top-K SAE (stage 2) on each, then sparse probing.
# Usage: STAGE1_RUNS="runs/stage1/a runs/stage1/b ..." bash scripts/downstream_compare.sh
# Stage-1 runs are only read (checkpoints/latest.pt); model-XX follows the STAGE1_RUNS order.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
: "${STAGE1_RUNS:?Set STAGE1_RUNS to space-separated stage-1 run directories}"
: "${ACTIVATION_MANIFEST:?Set ACTIVATION_MANIFEST}"
RUN_ROOT="${RUN_ROOT:-runs/downstream}"
PROBE_TASKS="${PROBE_TASKS:-data/probe/ag-news.jsonl}"
ACTIVATION_CACHE="${ACTIVATION_CACHE:-data/probe/ag-news-activations}"
DEVICE="${DEVICE:-cuda}"
checkpoints=()
mkdir -p "$RUN_ROOT"
: > "$RUN_ROOT/models.txt"
i=0
for run in $STAGE1_RUNS; do
  checkpoints+=("$run/checkpoints/latest.pt")
  printf "model-%02d\t%s\n" "$i" "$run" >> "$RUN_ROOT/models.txt"
  i=$((i + 1))
done

if [[ "${SKIP_STAGE2:-0}" != "1" ]]; then
  args=(--checkpoints "${checkpoints[@]}" --output "$RUN_ROOT/stage2"
    --config configs/stage2_topk.yaml --device "$DEVICE"
    --activation-manifest "$ACTIVATION_MANIFEST"
    --set "seed=${SEED:-42}" --set "steps=${STEPS:-10000}"
    --set "dictionary_size=${DICTIONARY_SIZE:-16384}" --set "k=${K:-64}")
  if [[ "${RESUME:-0}" == "1" ]]; then args+=(--resume); fi
  sj-stage2 sweep "${args[@]}"
fi

if [[ ! -f "$PROBE_TASKS" ]]; then
  sj-probe prepare --datasets fancyzhx/ag_news --train-size 4000 --test-size 1000 \
    --seed 42 --output "$PROBE_TASKS"
fi
stage2_ckpts=()
for ((j = 0; j < i; j++)); do
  stage2_ckpts+=("$RUN_ROOT/stage2/$(printf 'model-%02d' "$j")/checkpoints/latest.pt")
done
sj-probe collect --tasks "$PROBE_TASKS" --checkpoint "${stage2_ckpts[0]}" \
  --output "$ACTIVATION_CACHE" --device "$DEVICE" --batch-size "${LLM_BATCH_SIZE:-8}"
sj-probe evaluate --tasks "$PROBE_TASKS" --activations "$ACTIVATION_CACHE" \
  --checkpoints "${stage2_ckpts[@]}" --output "$RUN_ROOT/probe" --device "$DEVICE"
git rev-parse HEAD > "$RUN_ROOT/code-commit.txt" 2>/dev/null || true
echo "Stage 2: $RUN_ROOT/stage2/report/validation.md   Probe: $RUN_ROOT/probe/summary.md   Models: $RUN_ROOT/models.txt"
