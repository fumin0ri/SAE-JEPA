"""Aggregate stage-1 evaluations into a FVU vs. Gaussianization report.

The lowest-SIGReg checkpoint is deliberately *not* declared the winner: the
report marks the Pareto front over (FVU, held-out SIGReg) and leaves the
choice of stage-2 candidates to the experimenter.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .data import write_json


COLUMNS = [
    "run_name",
    "sigreg_weight",
    "seed",
    "step",
    "split",
    "positions",
    "reconstruction/fvu",
    "reconstruction/mse",
    "gaussian/heldout_sigreg",
    "reference/heldout_sigreg",
    "gaussian/diagnostic_sigreg",
    "reference/diagnostic_sigreg",
    "gaussian/diagnostic_w2_sq",
    "reference/diagnostic_w2_sq",
    "gaussian/diagnostic_quantile_abs_dev_mean",
    "reference/diagnostic_quantile_abs_dev_mean",
    "gaussian/mean_sq_per_dim",
    "gaussian/variance_mean",
    "gaussian/variance_abs_dev_mean",
    "gaussian/cov_fro_dev",
    "reference/cov_fro_dev",
    "gaussian/cov_effective_rank",
    "path",
]


def collect(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(run_root.rglob("eval-*.json")):
        values = json.loads(path.read_text(encoding="utf-8"))
        values["path"] = str(path.relative_to(run_root))
        rows.append(values)
    return rows


def pareto_front(rows: list[dict[str, Any]], x: str, y: str) -> set[int]:
    front = set()
    for i, a in enumerate(rows):
        if a.get(x) is None or a.get(y) is None:
            continue
        dominated = any(
            b.get(x) is not None
            and b.get(y) is not None
            and b[x] <= a[x]
            and b[y] <= a[y]
            and (b[x] < a[x] or b[y] < a[y])
            for j, b in enumerate(rows)
            if j != i
        )
        if not dominated:
            front.add(i)
    return front


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return "" if value is None else str(value)


def write_report(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in sorted({row["split"] for row in rows}):
        subset = sorted(
            (row for row in rows if row["split"] == split),
            key=lambda r: (r.get("sigreg_weight", 0.0), r.get("seed", 0), r.get("step", 0)),
        )
        front = pareto_front(subset, "reconstruction/fvu", "gaussian/heldout_sigreg")
        for index, row in enumerate(subset):
            row["pareto"] = index in front
        columns = COLUMNS + ["pareto"]
        with (output_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(subset)
        write_json(output_dir / f"{split}.json", subset)
        short = [
            ("lambda", "sigreg_weight"),
            ("seed", "seed"),
            ("step", "step"),
            ("FVU", "reconstruction/fvu"),
            ("SIGReg", "gaussian/heldout_sigreg"),
            ("SIGReg ref", "reference/heldout_sigreg"),
            ("diag W2^2", "gaussian/diagnostic_w2_sq"),
            ("W2^2 ref", "reference/diagnostic_w2_sq"),
            ("cov dev", "gaussian/cov_fro_dev"),
            ("cov ref", "reference/cov_fro_dev"),
            ("Pareto", "pareto"),
        ]
        lines = [
            f"# Stage 1 dense representations: {split}",
            "",
            "Pareto = not dominated on (FVU, held-out SIGReg). `ref` columns are the same "
            "statistic on exact N(0, I) samples with identical N, batches and projections.",
            "",
            "| " + " | ".join(name for name, _ in short) + " |",
            "|" + "---|" * len(short),
        ]
        for row in subset:
            lines.append("| " + " | ".join(_format(row.get(key)) for _, key in short) + " |")
        (output_dir / f"{split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        _plot(subset, output_dir / f"{split}_tradeoff.png")


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    points = [
        r for r in rows
        if r.get("reconstruction/fvu") is not None and r.get("gaussian/heldout_sigreg") is not None
    ]
    if not points:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.scatter(
        [r["reconstruction/fvu"] for r in points],
        [r["gaussian/heldout_sigreg"] for r in points],
        c=["tab:orange" if r.get("pareto") else "tab:blue" for r in points],
    )
    for r in points:
        ax.annotate(f"λ={r.get('sigreg_weight')}", (r["reconstruction/fvu"], r["gaussian/heldout_sigreg"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    reference = [r["reference/heldout_sigreg"] for r in points if "reference/heldout_sigreg" in r]
    if reference:
        ax.axhline(sum(reference) / len(reference), color="gray", ls="--", lw=1, label="Gaussian reference")
        ax.legend(fontsize=8)
    ax.set_xlabel("FVU (residual space)")
    ax.set_ylabel("held-out SIGReg")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate stage-1 evaluation JSON files")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    root = Path(args.run_root)
    rows = collect(root)
    if not rows:
        raise SystemExit(f"no eval-*.json under {root}")
    write_report(rows, Path(args.output_dir) if args.output_dir else root / "report")


if __name__ == "__main__":
    main()
