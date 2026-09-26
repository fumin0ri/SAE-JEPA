#!/usr/bin/env bash
# Tasks must already be prepared; collect once, compare existing stage-2 SAEs.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
: "${PROBE_TASKS:?Set PROBE_TASKS to a labeled train/validation/test JSONL}"
STAGE2_ROOT="${STAGE2_ROOT:-runs/stage2-pilot}"
ACTIVATION_CACHE="${ACTIVATION_CACHE:-data/probe/activations}"
RUN_ROOT="${RUN_ROOT:-runs/sparse-probing}"
DEVICE="${DEVICE:-cuda}"
sj-probe collect --tasks "$PROBE_TASKS" \
  --checkpoint "$STAGE2_ROOT/model-00/checkpoints/latest.pt" \
  --output "$ACTIVATION_CACHE" --device "$DEVICE" --batch-size "${LLM_BATCH_SIZE:-8}"
sj-probe evaluate --tasks "$PROBE_TASKS" --activations "$ACTIVATION_CACHE" \
  --checkpoints "$STAGE2_ROOT/model-00/checkpoints/latest.pt" \
    "$STAGE2_ROOT/model-01/checkpoints/latest.pt" \
    "$STAGE2_ROOT/model-02/checkpoints/latest.pt" \
  --output "$RUN_ROOT" --device "$DEVICE" "$@"
git rev-parse HEAD > "$RUN_ROOT/code-commit.txt"
python -m pip freeze > "$RUN_ROOT/python-environment.txt"
