#!/usr/bin/env bash
# Multi-seed downstream comparison: downstream_compare.sh once per stage-2 seed,
# sharing one probe task file and one LLM activation cache, then a seed summary.
# Seeds whose probe summary exists are skipped; a partial stage 2 is resumed.
# Usage: ACTIVATION_MANIFEST=/path/manifest.json bash scripts/downstream_seeds.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
: "${ACTIVATION_MANIFEST:?Set ACTIVATION_MANIFEST}"
STAGE1_RUNS="${STAGE1_RUNS:-runs/stage1/ablation-lambda-0-beta-0-dim-256-100k-decay50
runs/stage1/gaussian-sweep-m-1024-lambda-0.01-beta-100-100k-decay50
runs/stage1/split-half-corr-warmup5k-m-1024-lambda-0.01-beta-300-100k-decay50
runs/stage1/split-half-corr-warmup20k-m-1024-lambda-0.01-beta-1000-100k-decay50}"
SEEDS="${SEEDS:-42 43 44}"
RUN_ROOT="${RUN_ROOT:-runs/downstream-seeds}"
PROBE_TASKS="${PROBE_TASKS:-data/probe/saebench-8.jsonl}"
PROBE_DATASETS="${PROBE_DATASETS:-}"  # empty: all SAEBench sparse-probing datasets
ACTIVATION_CACHE="${ACTIVATION_CACHE:-$RUN_ROOT/probe-cache}"

for run in $STAGE1_RUNS; do
  [[ -f "$run/checkpoints/latest.pt" ]] || { echo "missing $run/checkpoints/latest.pt" >&2; exit 1; }
done
if [[ ! -f "$PROBE_TASKS" ]]; then
  prepare_args=(--train-size 4000 --test-size 1000 --seed 42 --output "$PROBE_TASKS")
  if [[ -n "$PROBE_DATASETS" ]]; then prepare_args+=(--datasets $PROBE_DATASETS); fi
  sj-probe prepare "${prepare_args[@]}"
fi

for seed in $SEEDS; do
  seed_root="$RUN_ROOT/seed-$seed"
  if [[ -f "$seed_root/probe/summary.md" ]]; then
    echo "Skipping seed $seed (finished)"
    continue
  fi
  echo "Seed $seed"
  resume=0
  if [[ -f "$seed_root/stage2/comparison.json" ]]; then resume=1; fi
  skip_stage2=0
  if [[ -f "$seed_root/stage2/report/validation.md" ]]; then skip_stage2=1; fi
  # A partial probe directory would be refused as non-empty; start it over.
  rm -rf "$seed_root/probe"
  STAGE1_RUNS="$STAGE1_RUNS" SEED="$seed" RUN_ROOT="$seed_root" RESUME="$resume" \
    SKIP_STAGE2="$skip_stage2" PROBE_TASKS="$PROBE_TASKS" ACTIVATION_CACHE="$ACTIVATION_CACHE" \
    bash scripts/downstream_compare.sh
done

python scripts/downstream_seed_summary.py "$RUN_ROOT" --seeds $SEEDS
