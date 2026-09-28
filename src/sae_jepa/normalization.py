"""Train-only input statistics.

Stage 1 maps a residual ``h`` to ``x = (h - mu) / s`` where ``mu`` is the train
mean vector and ``s`` is one scalar shared by all coordinates:

    s^2 = E_train ||h - mu||_2^2 / d

Only mean and overall scale are removed; correlations and the per-coordinate
variance profile are left for the encoder.  PCA whitening (``sj-fit-pca``) and
the raw input (``sj-make-raw-frontend``) are separate stage-2 comparison
front-ends, never inside the dense model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .data import DataSource, parse_entry, torch_load


NORMALIZATION_FORMAT = "sae-jepa-normalization-v1"
PCA_FORMAT = "sae-jepa-pca-whitening-v1"
FRONTEND_FORMAT = "sae-jepa-frontend-v1"


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


def resolve_pca_epsilon(eigenvalues: torch.Tensor, epsilon: float | None, relative_epsilon: float) -> float:
    """Absolute ``epsilon`` if given, else ``relative_epsilon`` x mean eigenvalue.

    ``x`` has mean squared norm ``d`` per token, so the mean eigenvalue is about 1
    and an absolute 1e-6 would amplify the weakest directions ~1000x.
    """
    if epsilon is not None:
        if not epsilon > 0:
            raise ValueError("epsilon must be positive")
        return float(epsilon)
    if not relative_epsilon > 0:
        raise ValueError("relative_epsilon must be positive")
    return float(relative_epsilon * eigenvalues.clamp_min(0).mean())


def _sorted_eigh(covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    return eigenvalues[order], eigenvectors[:, order]


def pca_diagnostics(
    eigenvalues: torch.Tensor,
    epsilon: float,
    halves: tuple[torch.Tensor, torch.Tensor] | None = None,
    ks: tuple[int, ...] = (1, 4, 16, 64, 256, 1024),
) -> dict[str, Any]:
    """Spectrum summary and split-half stability of the leading subspaces.

    ``subspace_overlap[k] = ||U_a[:, :k]^T U_b[:, :k]||_F^2 / k`` (1 = identical
    top-k subspaces in the two halves, ~k/d for unrelated ones).
    """
    lam = eigenvalues.double().clamp_min(0)
    total = float(lam.sum())
    d = lam.numel()
    ks = tuple(k for k in ks if k <= d)
    gain = (lam + epsilon).rsqrt()
    result = {
        "epsilon": epsilon,
        "trace": total,
        "participation_ratio": total**2 / max(float(lam.square().sum()), 1e-300),
        "largest_eigenvalue": float(lam[0]),
        "smallest_eigenvalue": float(lam[-1]),
        "condition_number": float((lam[0] + epsilon) / (lam[-1] + epsilon)),
        "maximum_whitening_gain": float(gain.max()),
        "eigenvalues_below_epsilon": int((lam < epsilon).sum()),
        "cumulative_variance": {str(k): float(lam[:k].sum()) / max(total, 1e-300) for k in ks},
        # Mean over whitened coordinates of the share of their unit variance
        # that is signal rather than the epsilon floor.
        "mean_signal_fraction": float((lam / (lam + epsilon)).mean()),
    }
    if halves is not None:
        (_, vectors_a), (_, vectors_b) = (_sorted_eigh(c) for c in halves)
        result["subspace_overlap"] = {
            str(k): float((vectors_a[:, :k].T @ vectors_b[:, :k]).square().sum()) / k for k in ks
        }
    return result


@torch.no_grad()
def pca_train_chunks(source, maximum_positions, batch_size, sample_seed, progress=False):
    """Read sampled train rows in shard/row order, retaining legacy half labels.

    Keep randperm's exact sample for compatibility with existing experiments.
    Only the I/O and summation order changes; do not reuse this for SIGReg
    evaluation, which requires shuffled batches.
    """
    if batch_size < 1 or maximum_positions < 0:
        raise ValueError("batch_size must be positive and maximum_positions nonnegative")
    entries = source.paths("train")
    counts = [int(source.sequence_counts(entry).sum()) for entry in entries]
    if maximum_positions == 0:
        # Bound GPU memory even when using the entire training set.
        for index, (entry, count) in enumerate(zip(entries, counts)):
            for start in range(0, count, batch_size):
                rows = source.gather(entry, torch.arange(start, min(start + batch_size, count)))
                yield rows, torch.full((len(rows),), index % 2, dtype=torch.long)
        return
    n_batches = min(sum(counts) // batch_size, maximum_positions // batch_size)
    if n_batches < 2:
        raise ValueError("maximum_positions and train split must cover at least two batches")
    if progress:
        print("PCA sampling: building legacy-compatible train permutation", flush=True)
    order = torch.randperm(sum(counts), generator=torch.Generator().manual_seed(sample_seed))
    # Clone releases the full-train backing storage after selection.
    selected = order[:n_batches * batch_size].clone()
    del order
    selected, permutation = selected.sort()
    half = (permutation // batch_size) % 2
    del permutation
    offset = 0
    for entry, count in zip(entries, counts):
        begin, end = torch.searchsorted(selected, torch.tensor([offset, offset + count])).tolist()
        for start in range(begin, end, batch_size):
            stop = min(start + batch_size, end)
            yield source.gather(entry, selected[start:stop] - offset), half[start:stop]
        offset += count


@torch.no_grad()
def fit_pca_whitening(
    source: DataSource,
    stats: dict[str, Any],
    epsilon: float | None = None,
    maximum_positions: int = 0,
    progress: bool = False,
    relative_epsilon: float = 1e-3,
    sample_seed: int = 92001,
    batch_size: int = 4096,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """PCA whitening of ``x = (h - mu)/s`` fitted on train positions only.

    ``maximum_positions > 0`` draws that many positions (rounded down to whole
    batches) uniformly without replacement from the whole train split with
    ``sample_seed``; 0 uses every train position.  The second moment about the
    train mean ``mu`` is accumulated in float64, and two halves (alternating
    batches / shards) are kept for a split-half stability check.  ``epsilon``
    is absolute when given, otherwise ``relative_epsilon`` x mean eigenvalue.
    """
    check_normalization_matches(stats, source)
    device = torch.device(device)
    mean = stats["mean"].double().to(device)
    scale = float(stats["scale"])
    d = mean.numel()
    halves = [torch.zeros(d, d, dtype=torch.float64, device=device) for _ in range(2)]
    counts = [0, 0]
    chunks = pca_train_chunks(source, maximum_positions, batch_size, sample_seed, progress)
    total = source.split_positions("train")
    if maximum_positions > 0:
        total = min(total // batch_size, maximum_positions // batch_size) * batch_size
    started = time.perf_counter()
    read_seconds = compute_seconds = 0.0
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    with tqdm(total=total, desc="PCA covariance (ordered reads)", unit="tokens", disable=not progress) as bar:
        iterator = iter(chunks)
        while True:
            tick = time.perf_counter()
            try:
                rows, half = next(iterator)
            except StopIteration:
                break
            read_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            x = (rows.to(device).double() - mean) / scale
            half = half.to(device)
            for label in (0, 1):
                part = x[half == label]
                if len(part):
                    halves[label].addmm_(part.T, part)
                    counts[label] += len(part)
            synchronize()
            compute_seconds += time.perf_counter() - tick
            bar.update(len(rows))
            bar.set_postfix(read_s=round(read_seconds, 1), compute_s=round(compute_seconds, 1))
    count = sum(counts)
    if min(counts) < 1:
        raise ValueError("PCA needs train positions in both halves")
    covariance = (halves[0] + halves[1]) / count
    if progress:
        print("PCA eigendecomposition: full covariance (1/3)", flush=True)
    synchronize()
    tick = time.perf_counter()
    eigenvalues, eigenvectors = _sorted_eigh(covariance)
    synchronize()
    eigh_seconds = time.perf_counter() - tick
    epsilon = resolve_pca_epsilon(eigenvalues, epsilon, relative_epsilon)
    if progress:
        print("PCA diagnostics: two split-half eigendecompositions (2/3, 3/3)", flush=True)
    tick = time.perf_counter()
    diagnostics = pca_diagnostics(
        eigenvalues, epsilon, (halves[0] / counts[0], halves[1] / counts[1])
    )
    diagnostics.update({"count": count, "half_counts": counts})
    synchronize()
    diagnostics["timing_seconds"] = {
        "sampling_and_read": read_seconds, "covariance_compute": compute_seconds,
        "full_eigh": eigh_seconds, "split_half_diagnostics": time.perf_counter() - tick,
        "total": time.perf_counter() - started,
    }
    return {
        "format": PCA_FORMAT,
        "mean": stats["mean"],
        "scale": scale,
        "eigenvalues": eigenvalues.float().cpu(),
        "eigenvectors": eigenvectors.float().cpu(),
        "epsilon": epsilon,
        "count": int(count),
        "sampling": {"maximum_positions": maximum_positions, "seed": sample_seed,
                     "read_order": "shard then ascending row; original half assignments preserved",
                     "batch_size": batch_size,
                     "method": "uniform without replacement over the train split"
                               if maximum_positions > 0 else "all train positions"},
        "diagnostics": diagnostics,
        "manifest_fingerprint": source.fingerprint,
        "shards": list(source.splits["train"]),
        "burn_in_excluded": source.burn_in,
        "convention": "y = U^T x / sqrt(lambda + epsilon), x = (h - mu)/s",
    }


def data_policy(source: DataSource, test_split: str, holdout_test_fraction: float,
                skip_burn_in: bool) -> dict[str, Any]:
    """The ``config.data`` subset stage 2 needs to rebuild the same DataSource."""
    return {"activation_manifest": str(source.manifest_path), "skip_burn_in": skip_burn_in,
            "skip_leading_positions": source.skip_leading_positions, "test_split": test_split,
            "holdout_test_fraction": holdout_test_fraction}


def make_frontend_file(kind: str, source: DataSource, stats: dict[str, Any],
                       policy: dict[str, Any], pca: dict[str, Any] | None = None) -> dict[str, Any]:
    """A raw / PCA stage-2 front-end with its data policy and statistics."""
    if kind not in {"raw", "pca"} or (kind == "pca") != (pca is not None):
        raise ValueError("front-end files are raw (no PCA) or pca (with PCA)")
    check_normalization_matches(stats, source)
    keys = ["mean", "scale", "count", "shards", "burn_in_excluded", "manifest_fingerprint"]
    state = {"format": FRONTEND_FORMAT, "kind": kind, "data": policy,
             "data_manifest": source.record(),
             "normalization": {key: stats[key] for key in keys if key in stats}}
    if pca is not None:
        state["pca"] = pca
    return state


def _source_arguments(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument("--activation-manifest", required=required)
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


def _frontend_arguments(parser: argparse.ArgumentParser) -> None:
    _source_arguments(parser, required=False)
    parser.add_argument(
        "--like",
        help="stage-1 dense checkpoint whose data policy and normalization statistics "
        "to copy (recommended: guarantees identical inputs across stage-2 candidates); "
        "--activation-manifest then only overrides a moved manifest path",
    )
    parser.add_argument("--normalization", help="sj-compute-normalization output (without --like)")
    parser.add_argument("--output", required=True)


def frontend_inputs_like(checkpoint: str | Path, manifest: str | None = None
                         ) -> tuple[DataSource, dict[str, Any], dict[str, Any]]:
    """Data source, statistics and data policy copied from a stage-1 checkpoint."""
    state = torch_load(checkpoint)
    data = dict(state["config"]["data"])
    if manifest:
        data["activation_manifest"] = manifest
    source = DataSource(data["activation_manifest"], skip_burn_in=data.get("skip_burn_in", True),
                        skip_leading_positions=data.get("skip_leading_positions", 0),
                        test_split=data.get("test_split", "auto"),
                        holdout_test_fraction=data.get("holdout_test_fraction", 0.5))
    record = source.record()
    if any(record[k] != state["data_manifest"][k] for k in ["fingerprint", "splits", "burn_in_excluded"]):
        raise ValueError("--like checkpoint data identity, splits or exclusions differ from the manifest")
    stats = state["normalization"]
    check_normalization_matches(stats, source)
    policy = data_policy(source, data.get("test_split", "auto"),
                         data.get("holdout_test_fraction", 0.5), data.get("skip_burn_in", True))
    return source, stats, policy


def _frontend_inputs(args: argparse.Namespace) -> tuple[DataSource, dict[str, Any], dict[str, Any]]:
    if args.like:
        return frontend_inputs_like(args.like, args.activation_manifest)
    if not args.activation_manifest or not args.normalization:
        raise SystemExit("give --like CHECKPOINT, or both --activation-manifest and --normalization")
    source = _source(args)
    stats = load_normalization(args.normalization, source)
    policy = data_policy(source, args.test_split, args.holdout_test_fraction, not args.no_skip_burn_in)
    return source, stats, policy


def _refuse_overwrite(path: str | Path) -> None:
    if Path(path).exists():
        raise SystemExit(f"{path} exists; choose a new output")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute train-only mean and scalar scale")
    _source_arguments(parser)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    stats = compute_train_statistics(_source(args), progress=True)
    save_normalization(stats, args.output)
    print(f"mean norm={stats['mean'].norm():.4f} scale={stats['scale']:.6f} n={stats['count']:,}")


def pca_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit a train-only PCA whitening stage-2 front-end")
    _frontend_arguments(parser)
    parser.add_argument("--epsilon", type=float, default=None,
                        help="absolute eigenvalue floor (overrides --relative-epsilon)")
    parser.add_argument("--relative-epsilon", type=float, default=1e-3,
                        help="eigenvalue floor as a fraction of the mean eigenvalue")
    parser.add_argument("--maximum-positions", type=int, default=2_097_152,
                        help="uniformly sampled train positions (0 = every train position)")
    parser.add_argument("--sample-seed", type=int, default=92001)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    _refuse_overwrite(args.output)
    source, stats, policy = _frontend_inputs(args)
    pca = fit_pca_whitening(source, stats, args.epsilon, args.maximum_positions, progress=True,
                            relative_epsilon=args.relative_epsilon, sample_seed=args.sample_seed,
                            batch_size=args.batch_size, device=args.device)
    save_normalization(make_frontend_file("pca", source, stats, policy, pca), args.output)
    summary = Path(args.output).with_suffix(".json")
    summary.write_text(json.dumps({"output": str(args.output), "data": policy,
                                   "sampling": pca["sampling"], "diagnostics": pca["diagnostics"]},
                                  indent=2), encoding="utf-8")
    d = pca["diagnostics"]
    print(f"PCA from {pca['count']:,} positions: epsilon={d['epsilon']:.3e} "
          f"condition={d['condition_number']:.3e} max gain={d['maximum_whitening_gain']:.1f} "
          f"participation ratio={d['participation_ratio']:.1f}; details in {summary}")


def raw_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write a raw (x = (h - mu)/s) stage-2 front-end")
    _frontend_arguments(parser)
    args = parser.parse_args(argv)
    _refuse_overwrite(args.output)
    source, stats, policy = _frontend_inputs(args)
    save_normalization(make_frontend_file("raw", source, stats, policy), args.output)


if __name__ == "__main__":
    main()
