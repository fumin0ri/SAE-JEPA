"""Train-only input statistics.

Stage 1 maps a residual ``h`` to ``x = (h - mu) / s`` where ``mu`` is the train
mean vector and ``s`` is one scalar shared by all coordinates:

    s^2 = E_train ||h - mu||_2^2 / d

Only mean and overall scale are removed; correlations and the per-coordinate
variance profile are left for the encoder.  PCA whitening is provided as a
separate comparison front-end (``sj-fit-pca``), never inside the dense model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .data import DataSource, parse_entry, torch_load


NORMALIZATION_FORMAT = "sae-jepa-normalization-v1"
PCA_FORMAT = "sae-jepa-pca-whitening-v1"


@torch.no_grad()
def compute_train_statistics(source: DataSource, progress: bool = False) -> dict[str, Any]:
    """Mean vector and scalar scale from the train split only.

    Shard statistics are merged with Chan's parallel update in float64 so that
    large residual means do not cancel catastrophically.
    """
    count = 0
    mean: torch.Tensor | None = None
    m2 = 0.0  # sum over positions of ||h - mean||^2
    paths = source.paths("train")
    for path in tqdm(paths, desc="train statistics", disable=not progress):
        rows = source.positions(path).double()
        n_b = len(rows)
        if n_b == 0:
            continue
        mean_b = rows.mean(dim=0)
        m2_b = float((rows - mean_b).square().sum())
        if mean is None:
            count, mean, m2 = n_b, mean_b, m2_b
            continue
        total = count + n_b
        delta = mean_b - mean
        mean = mean + delta * (n_b / total)
        m2 = m2 + m2_b + float(delta.square().sum()) * count * n_b / total
        count = total
    if mean is None or count < 2:
        raise ValueError("train split holds fewer than two positions")
    d = mean.numel()
    scale = max(m2 / count / d, 1e-24) ** 0.5
    return {
        "format": NORMALIZATION_FORMAT,
        "mean": mean.float(),
        "scale": float(scale),
        "count": int(count),
        "d_in": int(d),
        "split": "train",
        "shards": list(source.splits["train"]),
        "burn_in_excluded": source.burn_in,
        "manifest_fingerprint": source.fingerprint,
        "convention": "x = (h - mu) / s, s^2 = E_train ||h - mu||^2 / d (population)",
    }


def save_normalization(stats: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, out)


def load_normalization(
    path: str | Path, source: DataSource | None = None
) -> dict[str, Any]:
    stats = torch_load(path)
    if stats.get("format") != NORMALIZATION_FORMAT:
        raise ValueError(f"unsupported normalization file {path}")
    if stats.get("split") != "train":
        raise ValueError("normalization statistics must come from the train split")
    if source is not None:
        check_normalization_matches(stats, source)
    return stats


def check_normalization_matches(stats: dict[str, Any], source: DataSource) -> None:
    if stats["manifest_fingerprint"] != source.fingerprint:
        raise ValueError("normalization was computed for a different activation manifest")
    if list(stats["shards"]) != list(source.splits["train"]):
        raise ValueError("normalization shards differ from the train split")
    held_out = {
        parse_entry(entry)[0] for entry in source.splits["validation"] + source.splits["test"]
    }
    if held_out & {parse_entry(entry)[0] for entry in stats["shards"]}:
        raise ValueError("normalization statistics include held-out shards")
    if int(stats["burn_in_excluded"]) != source.burn_in:
        raise ValueError(
            "normalization excluded positions < "
            f"{int(stats['burn_in_excluded'])} but the data source excludes positions < "
            f"{source.burn_in}; recompute it with the same --skip-leading-positions "
            "(data.skip_leading_positions) and burn-in policy"
        )


def normalize(h: torch.Tensor, mean: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    return (h - mean) / scale


def denormalize(x: torch.Tensor, mean: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    return x * scale + mean


@torch.no_grad()
def fit_pca_whitening(
    source: DataSource,
    stats: dict[str, Any],
    epsilon: float = 1e-6,
    maximum_positions: int = 0,
    progress: bool = False,
) -> dict[str, Any]:
    """PCA whitening of ``x = (h - mu)/s`` fitted on train positions only."""
    check_normalization_matches(stats, source)
    mean = stats["mean"].double()
    scale = float(stats["scale"])
    d = mean.numel()
    second = torch.zeros(d, d, dtype=torch.float64)
    count = 0
    for path in tqdm(source.paths("train"), desc="PCA covariance", disable=not progress):
        x = (source.positions(path).double() - mean) / scale
        if maximum_positions > 0:
            x = x[: max(0, maximum_positions - count)]
        second += x.T @ x
        count += len(x)
        if maximum_positions > 0 and count >= maximum_positions:
            break
    covariance = second / max(count, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    return {
        "format": PCA_FORMAT,
        "mean": stats["mean"],
        "scale": scale,
        "eigenvalues": eigenvalues[order].float(),
        "eigenvectors": eigenvectors[:, order].float(),
        "epsilon": float(epsilon),
        "count": int(count),
        "manifest_fingerprint": source.fingerprint,
        "shards": list(source.splits["train"]),
        "convention": "y = U^T x / sqrt(lambda + epsilon), x = (h - mu)/s",
    }


def _source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--activation-manifest", required=True)
    parser.add_argument("--no-skip-burn-in", action="store_true")
    parser.add_argument(
        "--skip-leading-positions",
        type=int,
        default=0,
        help="drop token positions < k of every stored sequence (must match "
        "data.skip_leading_positions of the runs that use these statistics)",
    )
    parser.add_argument("--test-split", default="auto")
    parser.add_argument("--holdout-test-fraction", type=float, default=0.5)


def _source(args: argparse.Namespace) -> DataSource:
    return DataSource(
        args.activation_manifest,
        skip_burn_in=not args.no_skip_burn_in,
        skip_leading_positions=args.skip_leading_positions,
        test_split=args.test_split,
        holdout_test_fraction=args.holdout_test_fraction,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute train-only mean and scalar scale")
    _source_arguments(parser)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    stats = compute_train_statistics(_source(args), progress=True)
    save_normalization(stats, args.output)
    print(f"mean norm={stats['mean'].norm():.4f} scale={stats['scale']:.6f} n={stats['count']:,}")


def pca_main() -> None:
    parser = argparse.ArgumentParser(description="Fit train-only PCA whitening")
    _source_arguments(parser)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--maximum-positions", type=int, default=0)
    args = parser.parse_args()
    source = _source(args)
    stats = load_normalization(args.normalization, source)
    pca = fit_pca_whitening(source, stats, args.epsilon, args.maximum_positions, progress=True)
    save_normalization(pca, args.output)


if __name__ == "__main__":
    main()
