#!/usr/bin/env bash
# Re-evaluate every latest checkpoint under RUN_ROOT and rebuild the report.
set -euo pipefail

: "${RUN_ROOT:?Set RUN_ROOT}"
SPLITS="${SPLITS:-validation test}"
DEVICE="${DEVICE:-cuda}"

find "$RUN_ROOT" -path '*/checkpoints/latest.pt' | sort | while read -r checkpoint; do
  for split in $SPLITS; do
    sj-evaluate-dense --checkpoint "$checkpoint" --split "$split" --device "$DEVICE"
  done
done
sj-report-dense --run-root "$RUN_ROOT"
