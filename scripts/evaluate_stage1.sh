#!/usr/bin/env bash
# Re-evaluate every latest checkpoint under RUN_ROOT and rebuild the report.
# Works on checkpoints from earlier versions: new eval.* settings take their
# defaults, and EVAL_ARGS can override them, e.g.
#   EVAL_ARGS="--set eval.leading_positions=4 --set eval.outlier_fraction=0.01"
set -euo pipefail

: "${RUN_ROOT:?Set RUN_ROOT}"
SPLITS="${SPLITS:-validation test}"
DEVICE="${DEVICE:-cuda}"
EVAL_ARGS=(${EVAL_ARGS:-})

find "$RUN_ROOT" -path '*/checkpoints/latest.pt' | sort | while read -r checkpoint; do
  for split in $SPLITS; do
    sj-evaluate-dense --checkpoint "$checkpoint" --split "$split" --device "$DEVICE" \
      "${EVAL_ARGS[@]}"
  done
done
sj-report-dense --run-root "$RUN_ROOT"
