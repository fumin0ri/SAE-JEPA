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
    # outlier diagnostics (present for evaluations run with this version)
    "reconstruction/fvu_excluding_leading",
    "leading/sse_share",
    "gaussian/y_sqnorm_q0.5",
    "gaussian/y_sqnorm_q0.999",
    "gaussian/y_sqnorm_q1",
    "reference/y_sqnorm_q0.999",
    "reference/y_sqnorm_q1",
    "leading/y_sqnorm_mean",
    "nonleading/y_sqnorm_mean",
    "leading/x_sqnorm_mean",
    "nonleading/x_sqnorm_mean",
    "outliers/count",
    "outliers/leading_share",
    "outliers/y_sqnorm_share",
    "gaussian/excl_leading/cov_fro_dev",
    "gaussian/excl_leading/cov_eig_max",
    "gaussian/excl_leading/cov_effective_rank",
    "gaussian/trimmed/cov_fro_dev",
    "gaussian/trimmed/cov_eig_max",
    "gaussian/trimmed/cov_effective_rank",
    "reference/trimmed/cov_effective_rank",
    "path",
]

DIAGNOSTIC_TABLE = [
    ("lambda", "sigreg_weight"),
    ("FVU excl. lead", "reconstruction/fvu_excluding_leading"),
    ("lead SSE share", "leading/sse_share"),
    ("‖y‖²/d q0.5", "gaussian/y_sqnorm_q0.5"),
    ("q0.999", "gaussian/y_sqnorm_q0.999"),
    ("max", "gaussian/y_sqnorm_q1"),
    ("ref max", "reference/y_sqnorm_q1"),
    ("lead ‖y‖²/d", "leading/y_sqnorm_mean"),
    ("rest ‖y‖²/d", "nonleading/y_sqnorm_mean"),
    ("top: lead share", "outliers/leading_share"),
    ("top: ‖y‖² share", "outliers/y_sqnorm_share"),
    ("eff. rank", "gaussian/cov_effective_rank"),
    ("excl. lead", "gaussian/excl_leading/cov_effective_rank"),
    ("trimmed", "gaussian/trimmed/cov_effective_rank"),
    ("ref", "reference/trimmed/cov_effective_rank"),
    ("eig max", "gaussian/cov_eig_max"),
    ("excl. lead", "gaussian/excl_leading/cov_eig_max"),
    ("trimmed", "gaussian/trimmed/cov_eig_max"),
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


def write_report(
    rows: list[dict[str, Any]], output_dir: Path, run_root: Path | None = None
) -> None:
    """``run_root`` is where row paths are relative to (default: parent of output_dir)."""
    run_root = run_root or output_dir.parent
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
        diagnosed = [row for row in subset if "outliers/count" in row]
        if diagnosed:
            leading = diagnosed[0].get("diagnostic/leading_positions", 1)
            lines += [
                "",
                "## Outlier diagnostics",
                "",
                f"`lead` = token positions < {leading} within each stored sequence. "
                "`top` = the largest `outliers/count` samples by ‖y‖². `excl. lead` / "
                "`trimmed` = covariance without leading positions / without the top samples. "
                "Eigenvalue spectra: `*_spectra.png` and `eval-*-spectra.pt`.",
                "",
                "| " + " | ".join(name for name, _ in DIAGNOSTIC_TABLE) + " |",
                "|" + "---|" * len(DIAGNOSTIC_TABLE),
            ]
            for row in diagnosed:
                lines.append(
                    "| " + " | ".join(_format(row.get(key)) for _, key in DIAGNOSTIC_TABLE) + " |"
                )
        (output_dir / f"{split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        _plot(subset, output_dir / f"{split}_tradeoff.png")
        _plot_spectra(diagnosed, output_dir / f"{split}_spectra.png", root=run_root)


def _plot_spectra(rows: list[dict[str, Any]], path: Path, root: Path) -> None:
    """Sorted covariance eigenvalues (full / excl. leading / trimmed) per run."""
    rows = [row for row in rows if row.get("spectra_path")]
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch
    except ImportError:
        return
    kinds = ("full", "excl_leading", "trimmed")
    fig, axes = plt.subplots(1, len(kinds), figsize=(13, 3.8), sharey=True)
    reference_drawn = False
    for row in rows:
        spectra_file = (root / row["path"]).with_name(row["spectra_path"])
        if not spectra_file.exists():
            continue
        spectra = torch.load(spectra_file, map_location="cpu", weights_only=False)
        if "reference" in spectra and not reference_drawn:
            values = spectra["reference"].clamp_min(1e-12).numpy()
            for ax in axes:
                ax.plot(range(1, len(values) + 1), values, color="gray", lw=2, alpha=0.5,
                        label="N(0, I), same n")
            reference_drawn = True
        for ax, kind in zip(axes, kinds):
            if kind in spectra:
                values = spectra[kind].clamp_min(1e-12).numpy()
                ax.plot(range(1, len(values) + 1), values, lw=1.2,
                        label=f"λ={row.get('sigreg_weight')}")
    for ax, kind in zip(axes, kinds):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(kind)
        ax.set_xlabel("eigenvalue index")
    axes[0].set_ylabel("Cov(y) eigenvalue")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    write_report(rows, Path(args.output_dir) if args.output_dir else root / "report", run_root=root)


if __name__ == "__main__":
    main()
