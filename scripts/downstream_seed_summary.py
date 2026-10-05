"""Mean and spread of sparse-probe accuracy across stage-2 seeds.

Reads RUN_ROOT/seed-*/probe/comparison.json (written by sj-probe evaluate) and
RUN_ROOT/seed-*/models.txt, and writes RUN_ROOT/seed-summary.{json,md}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def stage1_runs(seed_root):
    lines = (seed_root / "models.txt").read_text(encoding="utf-8").splitlines()
    return [line.split("\t", 1)[1] for line in lines if line.strip()]


def mean_std(values):
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_root")
    p.add_argument("--seeds", nargs="+", required=True)
    p.add_argument("--split", default="validation")
    args = p.parse_args()
    root = Path(args.run_root)

    runs = None
    scores = {}  # (run, k, dataset or "macro") -> [accuracy per seed]
    for seed in args.seeds:
        seed_root = root / f"seed-{seed}"
        path = seed_root / "probe/comparison.json"
        if not path.exists():
            print(f"seed {seed}: no probe results, skipped")
            continue
        seed_runs = stage1_runs(seed_root)
        if runs is None:
            runs = seed_runs
        elif seed_runs != runs:
            raise ValueError(f"seed {seed}: stage-1 runs differ from earlier seeds")
        models = json.loads(path.read_text(encoding="utf-8"))["models"]
        if len(models) != len(runs):
            raise ValueError(f"seed {seed}: {len(models)} probe models for {len(runs)} stage-1 runs")
        for run, model in zip(runs, models):
            for k, result in model["aggregate"][args.split].items():
                scores.setdefault((run, k, "macro"), []).append(result["dataset_macro_accuracy"])
                for dataset, accuracy in result["per_dataset_accuracy"].items():
                    scores.setdefault((run, k, dataset), []).append(accuracy)
    if runs is None:
        raise SystemExit("no finished seeds")

    rows = [{"run": run, "k": int(k), "dataset": dataset, "seeds": len(values),
             "mean": mean_std(values)[0], "std": mean_std(values)[1], "values": values}
            for (run, k, dataset), values in scores.items()]
    (root / "seed-summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    ks = sorted({row["k"] for row in rows})
    datasets = sorted({row["dataset"] for row in rows} - {"macro"})
    lookup = {(row["run"], row["k"], row["dataset"]): row for row in rows}
    cell = lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}"
    lines = [f"# Sparse probing across seeds ({args.split})", "",
             f"Seeds with results: {lookup[(runs[0], ks[0], 'macro')]['seeds']}. Values: mean ± sample std.", "",
             "## Dataset macro accuracy", "",
             "| stage-1 run | " + " | ".join(f"k={k}" for k in ks) + " |",
             "|---|" + "---|" * len(ks)]
    lines += [f"| {Path(run).name} | " + " | ".join(cell(lookup[(run, k, 'macro')]) for k in ks) + " |"
              for run in runs]
    for k in ks:
        lines += ["", f"## Per dataset, k={k}", "",
                  "| stage-1 run | " + " | ".join(Path(d).name for d in datasets) + " |",
                  "|---|" + "---|" * len(datasets)]
        lines += [f"| {Path(run).name} | " + " | ".join(cell(lookup[(run, k, d)]) for d in datasets) + " |"
                  for run in runs]
    (root / "seed-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
