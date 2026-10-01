"""Paired scalar diagnostics; never retain activation vectors or draw randomness."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def _ranks(x):
    """Average ranks for ties, including constant vectors."""
    _, inverse, counts = torch.unique(x, sorted=True, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0).double()
    return (ends - (counts.double() - 1) / 2)[inverse]


def _correlation(x, y):
    x, y = x - x.mean(), y - y.mean()
    denominator = x.norm() * y.norm()
    return float((x @ y / denominator).clamp(-1, 1)) if denominator > 0 else None


def _stats(x):
    if not len(x):
        return None
    return {"count": len(x), "mean": float(x.mean()), **{
        f"q{q:g}": float(torch.quantile(x, q)) for q in (0., .5, .9, .99, .999, 1.)}}


class NormDiagnostics:
    def __init__(self, fraction=.01):
        if not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError("norm outlier fraction must be between 0 and 1")
        self.fraction = fraction
        self.rows = []
        self.dimensions = None

    def add(self, h, x, y, metadata, entries, *, masked_x=None, masked_y=None):
        vectors = {"h": h, "x": x, "y": y}
        if masked_x is not None:
            vectors.update(masked_x=masked_x, masked_y=masked_y)
        dimensions = {key: value.shape[1] for key, value in vectors.items()}
        if self.dimensions is not None and self.dimensions != dimensions:
            raise ValueError("norm diagnostic dimensions changed")
        self.dimensions = dimensions
        values = {}
        for key, value in vectors.items():
            sq = value.detach().double().square().mean(1).cpu()
            if not torch.isfinite(sq).all():
                raise ValueError("nonfinite norm diagnostic input/output")
            values[key + "_sqnorm_per_dim"] = sq.tolist()
        for i in range(len(h)):
            row = {"sample_index": len(self.rows), "entry": int(metadata["entry"][i]),
                   "shard": entries[int(metadata["entry"][i])],
                   "sequence": int(metadata["sequence"][i]), "position": int(metadata["position"][i])}
            row.update({key: value[i] for key, value in values.items()})
            for output, input_ in [("y", "x"), ("masked_y", "masked_x")]:
                if output + "_sqnorm_per_dim" in row:
                    denominator = row[input_ + "_sqnorm_per_dim"]
                    row[output + "_over_" + input_ + "_energy_ratio"] = (
                        row[output + "_sqnorm_per_dim"] / denominator if denominator > 0 else None)
            self.rows.append(row)

    def summary(self):
        if not self.rows:
            raise ValueError("no norm diagnostic samples")
        columns = {key: torch.tensor([r[key] for r in self.rows], dtype=torch.float64)
                   for key in self.rows[0] if key.endswith("_sqnorm_per_dim")}
        n = len(self.rows)
        k = min(math.ceil(n * self.fraction), n - 1)
        if k < 1:
            raise ValueError("norm diagnostics require at least two samples")
        # Stable sort gives deterministic sample-index tie breaking.
        orders = {key: torch.argsort(x, descending=True, stable=True) for key, x in columns.items()}
        result = {"count": n, "dimensions": self.dimensions, "outlier_fraction": self.fraction,
                  "outlier_count": k, "norm_definition": "sum(vector**2) / vector_dimension",
                  "energy_ratio_definition": "output mean-square / corresponding normalized input mean-square; null at zero input",
                  "tie_breaking": "lower sample_index first for equal norms; Spearman uses average ranks",
                  "columns": {key: _stats(x) for key, x in columns.items()}, "views": {}}
        for view in ("y", "masked_y"):
            key = view + "_sqnorm_per_dim"
            if key not in columns:
                continue
            y = columns[key]
            selected = torch.zeros(n, dtype=torch.bool)
            selected[orders[key][:k]] = True
            comparisons = {}
            for input_ in ("h", "x", "masked_x"):
                other = input_ + "_sqnorm_per_dim"
                if other not in columns:
                    continue
                x = columns[other]
                overlap = int(selected[orders[other][:k]].sum())
                comparisons[input_] = {"pearson": _correlation(x, y),
                    "spearman": _correlation(_ranks(x), _ranks(y)),
                    "input_top_overlap_count": overlap, "input_top_overlap_fraction": overlap / k}
            input_ = "x" if view == "y" else "masked_x"
            ratio_key = view + "_over_" + input_ + "_energy_ratio"
            ratios = [r[ratio_key] for r in self.rows if r[ratio_key] is not None]
            result["views"][view] = {"input_comparisons": comparisons,
                "energy_ratio": _stats(torch.tensor(ratios, dtype=torch.float64)),
                "undefined_energy_ratio_count": n - len(ratios),
                "output_top_energy_fraction": float(y[selected].sum() / y.sum()) if y.sum() > 0 else None,
                "groups": {label: {c: _stats(v[selection]) for c, v in columns.items()}
                           for label, selection in [("output_top", selected), ("remaining", ~selected)]},
                "top_samples": [self.rows[i] for i in orders[key][:min(k, 32)].tolist()]}
        return result

    def write(self, output: Path, provenance):
        summary_path = output.with_name(output.stem + "-norm-summary.json")
        samples_path = output.with_name(output.stem + "-norm-samples.jsonl")
        result = self.summary()
        result["provenance"] = provenance
        result["samples_path"] = samples_path.name
        output.parent.mkdir(parents=True, exist_ok=True)
        with samples_path.open("w", encoding="utf-8") as f:
            for row in self.rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return {"summary_path": summary_path.name, "samples_path": samples_path.name}
