"""Held-out evaluation of a dense stage-1 representation on two axes.

Information retention (original residual space):
    mse   = sum ||h_hat - h||^2 / (n d)
    fvu   = sum ||h_hat - h||^2 / sum ||h - mean_eval(h)||^2
    (``mean_eval`` is the mean of the whole evaluated split, not per batch.)

Gaussianization of y:
    * held-out SIGReg on fixed validation projections (same N as training);
    * mean shift, per-dimension variance;
    * (detailed) covariance deviation ||Cov(y) - I||_F / sqrt(d) and spectrum;
    * (detailed) W2^2 and quantiles on *diagnostic* projections never used by
      any SIGReg computation;
    * (detailed) outlier diagnostics: per-sample ||y||^2/d distribution, where
      the largest samples sit in their sequences, and covariance without
      leading token positions and without the top-||y|| samples.  Random-
      projection SIGReg is nearly blind to rare extreme samples and to
      trace-preserving low-rank collapse, so these separate the two.
Every Gaussianization statistic is paired with a reference value obtained by
running the identical computation on exact N(0, I) samples with the same
number of batches, batch size, and projections, so finite-sample error is
visible.  Dense-model-only: there are no dead-feature or firing-rate metrics.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from .config import EvalConfig, SIGRegConfig, config_from_dict, parse_override, update_config
from .data import DataSource, eval_batches, write_json
from .models import ARCHITECTURE_ID, DenseSIGRegAE, build_model
from .sigreg import (
    epps_pulley_sigreg,
    fixed_projections,
    gaussian_expected_value,
    integration_grid,
)


def _autocast(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda" and amp_dtype == "bfloat16"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def projected_w2_squared(u: torch.Tensor) -> torch.Tensor:
    """Per-column W2^2 between the empirical law of ``u[:, m]`` and N(0, 1).

    Uses the sorted sample against Gaussian quantiles at (i - 1/2)/n.
    """
    n = u.shape[0]
    levels = (torch.arange(n, device=u.device, dtype=torch.float64) + 0.5) / n
    gaussian = torch.special.ndtri(levels)[:, None]
    ordered = torch.sort(u.double(), dim=0).values
    return (ordered - gaussian).square().mean(dim=0)


class _Moments:
    def __init__(self, d: int, device: torch.device, covariance: bool):
        self.n = 0
        self.sum = torch.zeros(d, dtype=torch.float64, device=device)
        self.sum_sq = torch.zeros(d, dtype=torch.float64, device=device)
        self.second = (
            torch.zeros(d, d, dtype=torch.float64, device=device) if covariance else None
        )

        self.eigenvalues: torch.Tensor | None = None

    def add(self, y: torch.Tensor, sign: float = 1.0) -> None:
        y = y.double()
        self.n += int(sign) * len(y)
        self.sum += sign * y.sum(0)
        self.sum_sq += sign * y.square().sum(0)
        if self.second is not None:
            self.second += sign * (y.T @ y)

    def without(self, removed: torch.Tensor) -> "_Moments":
        """Moments of the same samples with the rows ``removed`` taken out."""
        trimmed = _Moments.__new__(_Moments)
        trimmed.n = self.n
        trimmed.sum = self.sum.clone()
        trimmed.sum_sq = self.sum_sq.clone()
        trimmed.second = None if self.second is None else self.second.clone()
        trimmed.eigenvalues = None
        if len(removed):
            trimmed.add(removed, sign=-1.0)
        return trimmed

    def summary(self, prefix: str) -> dict[str, float]:
        n = max(self.n, 1)
        mean = self.sum / n
        variance = self.sum_sq / n - mean.square()
        d = mean.numel()
        out = {
            f"{prefix}mean_sq_per_dim": float(mean.square().mean()),
            f"{prefix}mean_abs_max": float(mean.abs().max()),
            f"{prefix}variance_mean": float(variance.mean()),
            f"{prefix}variance_abs_dev_mean": float((variance - 1).abs().mean()),
            f"{prefix}variance_min": float(variance.min()),
            f"{prefix}variance_max": float(variance.max()),
        }
        if self.second is not None:
            covariance = self.second / n - torch.outer(mean, mean)
            identity = torch.eye(d, dtype=covariance.dtype, device=covariance.device)
            off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
            self.eigenvalues = eigenvalues.flip(0).float().cpu()
            p = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
            out.update(
                {
                    f"{prefix}cov_fro_dev": float(
                        (covariance - identity).norm() / math.sqrt(d)
                    ),
                    f"{prefix}cov_offdiag_rms": float(
                        off_diagonal.square().sum().div(d * (d - 1)).sqrt()
                    ),
                    f"{prefix}cov_eig_min": float(eigenvalues.min()),
                    f"{prefix}cov_eig_max": float(eigenvalues.max()),
                    f"{prefix}cov_participation_ratio": float(
                        eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-30)
                    ),
                    f"{prefix}cov_effective_rank": float(
                        torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum())
                    ),
                }
            )
        return out


class _ProjectionDiagnostics:
    """SIGReg, W2^2 and quantiles on one fixed projection set."""

    def __init__(
        self,
        projections: torch.Tensor,
        t: torch.Tensor,
        weights: torch.Tensor,
        scale_by_batch_size: bool,
        quantiles: tuple[float, ...] | None,
        maximum_quantile_samples: int,
        w2: bool,
    ):
        self.projections = projections
        self.t = t
        self.weights = weights
        self.scale_by_batch_size = scale_by_batch_size
        self.quantiles = quantiles
        self.maximum_quantile_samples = maximum_quantile_samples
        self.w2 = w2
        self.sigreg: list[float] = []
        self.w2_values: list[float] = []
        self.samples: list[torch.Tensor] = []
        self.sample_count = 0

    def add(self, y: torch.Tensor) -> None:
        projections = self.projections.to(y.device)
        self.sigreg.append(
            float(
                epps_pulley_sigreg(
                    y, projections, self.t, self.weights, self.scale_by_batch_size
                )
            )
        )
        if not (self.w2 or self.quantiles):
            return
        u = y.float() @ projections
        if self.w2:
            self.w2_values.append(float(projected_w2_squared(u).mean()))
        if self.quantiles and self.sample_count < self.maximum_quantile_samples:
            keep = u[: self.maximum_quantile_samples - self.sample_count]
            self.samples.append(keep.cpu())
            self.sample_count += len(keep)

    def summary(self, prefix: str) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.sigreg:
            out[f"{prefix}sigreg"] = sum(self.sigreg) / len(self.sigreg)
        if self.w2_values:
            out[f"{prefix}w2_sq"] = sum(self.w2_values) / len(self.w2_values)
        if self.quantiles and self.samples:
            u = torch.cat(self.samples).double()
            levels = torch.tensor(self.quantiles, dtype=torch.float64)
            empirical = torch.quantile(u, levels, dim=0)  # [Q, M]
            target = torch.special.ndtri(levels)[:, None]
            deviation = empirical - target
            for q, row in zip(self.quantiles, deviation):
                out[f"{prefix}quantile_{q:g}_abs_dev"] = float(row.abs().mean())
                out[f"{prefix}quantile_{q:g}_signed_dev"] = float(row.mean())
            out[f"{prefix}quantile_abs_dev_mean"] = float(deviation.abs().mean())
        return out


class _TopRows:
    """The ``capacity`` rows with the largest score seen so far (and their ids)."""

    def __init__(self, capacity: int, d: int, device: torch.device):
        self.capacity = capacity
        self.rows = torch.empty(0, d, device=device)
        self.scores = torch.empty(0, device=device)
        self.ids = torch.empty(0, dtype=torch.long, device=device)

    def add(self, rows: torch.Tensor, scores: torch.Tensor, ids: torch.Tensor) -> None:
        rows = torch.cat([self.rows, rows.float()])
        scores = torch.cat([self.scores, scores.float()])
        ids = torch.cat([self.ids, ids.to(scores.device)])
        keep = torch.topk(scores, min(self.capacity, len(scores))).indices
        self.rows, self.scores, self.ids = rows[keep], scores[keep], ids[keep]

    def top(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        order = torch.argsort(self.scores, descending=True)[:k]
        return self.rows[order], self.ids[order]


POSITION_BINS = ((0, 1), (1, 4), (4, 16), (16, 64), (64, 256), (256, None))


class _OutlierDiagnostics:
    """Where do heavy tails of y come from, and what do they do to covariance?

    * per-sample ||y||^2/d, ||x||^2/d (normalized input) and reconstruction
      error, with the token position of every sample;
    * statistics restricted to non-leading positions (``position >=
      leading_positions``; position 0 is the first token of each stored
      sequence, where Pythia residuals are often extreme);
    * covariance after removing the top ``outlier_fraction`` of samples by
      ||y||^2 (exact: their contribution is subtracted from the full moments);
    * a table of the largest-||y|| samples with their shard, sequence and
      position.
    The Gaussian reference runs through the same subsets.
    """

    def __init__(self, d_latent: int, device: torch.device, cfg: EvalConfig, entries: list[str]):
        self.cfg = cfg
        self.entries = entries
        self.n = 0
        self.y_sq: list[torch.Tensor] = []
        self.reference_sq: list[torch.Tensor] = []
        self.x_sq: list[torch.Tensor] = []
        self.error: list[torch.Tensor] = []
        self.meta: dict[str, list[torch.Tensor]] = {"entry": [], "sequence": [], "position": []}
        self.nonleading = _Moments(d_latent, device, covariance=True)
        self.reference_nonleading = _Moments(d_latent, device, covariance=True)
        self.top = _TopRows(cfg.outlier_buffer, d_latent, device)
        self.reference_top = _TopRows(cfg.outlier_buffer, d_latent, device)
        self.sse = {"leading": 0.0, "rest": 0.0}
        self.sum_h_rest = None
        self.sum_h_sq_rest = 0.0
        self.n_rest = 0

    def add(self, h, h_hat, x, x_hat, y, gaussian, meta) -> None:
        ids = torch.arange(self.n, self.n + len(y), device=y.device)
        self.n += len(y)
        y_sq = y.square().mean(1)
        reference_sq = gaussian.square().mean(1)
        self.y_sq.append(y_sq.cpu())
        self.reference_sq.append(reference_sq.cpu())
        self.x_sq.append(x.float().square().mean(1).cpu())
        self.error.append((x_hat - x).float().square().mean(1).cpu())
        for key in self.meta:
            self.meta[key].append(meta[key].cpu())
        rest = (meta["position"] >= self.cfg.leading_positions).to(y.device)
        self.nonleading.add(y[rest])
        self.reference_nonleading.add(gaussian[rest])
        self.top.add(y, y_sq, ids)
        self.reference_top.add(gaussian, reference_sq, ids)
        sample_sse = (h_hat.double() - h.double()).square().sum(1)
        self.sse["leading"] += float(sample_sse[~rest].sum())
        self.sse["rest"] += float(sample_sse[rest].sum())
        h_rest = h[rest].double()
        self.sum_h_rest = h_rest.sum(0) if self.sum_h_rest is None else self.sum_h_rest + h_rest.sum(0)
        self.sum_h_sq_rest += float(h_rest.square().sum())
        self.n_rest += int(rest.sum())

    def summary(
        self, full: _Moments, reference_full: _Moments
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        cfg = self.cfg
        y_sq = torch.cat(self.y_sq).double()
        x_sq = torch.cat(self.x_sq).double()
        error = torch.cat(self.error).double()
        meta = {key: torch.cat(values) for key, values in self.meta.items()}
        position = meta["position"]
        leading = position < cfg.leading_positions
        out: dict[str, Any] = {"diagnostic/leading_positions": cfg.leading_positions}

        levels = torch.tensor(cfg.norm_quantiles, dtype=torch.float64)
        reference_sq = torch.cat(self.reference_sq).double()
        for name, values in (("gaussian/y_sqnorm", y_sq), ("reference/y_sqnorm", reference_sq),
                             ("input/x_sqnorm", x_sq), ("reconstruction/sample_error", error)):
            for q, value in zip(cfg.norm_quantiles, torch.quantile(values, levels)):
                out[f"{name}_q{q:g}"] = float(value)
            out[f"{name}_mean"] = float(values.mean())

        count = int(leading.sum())
        out["leading/count"] = count
        out["leading/fraction"] = count / max(self.n, 1)
        for name, values in (("y_sqnorm", y_sq), ("x_sqnorm", x_sq), ("sample_error", error)):
            out[f"leading/{name}_mean"] = float(values[leading].mean()) if count else None
            out[f"nonleading/{name}_mean"] = float(values[~leading].mean()) if count < self.n else None
        total_sse = self.sse["leading"] + self.sse["rest"]
        out["leading/sse_share"] = self.sse["leading"] / max(total_sse, 1e-30)
        if self.n_rest:
            mean_rest = self.sum_h_rest / self.n_rest
            variation = self.sum_h_sq_rest - self.n_rest * float(mean_rest.square().sum())
            out["reconstruction/fvu_excluding_leading"] = self.sse["rest"] / max(variation, 1e-30)

        k = min(max(1, math.ceil(cfg.outlier_fraction * self.n)), cfg.outlier_buffer)
        out["outliers/count"] = k
        top_rows, top_ids = self.top.top(k)
        top_ids = top_ids.cpu()
        top_leading = leading[top_ids]
        out["outliers/leading_share"] = float(top_leading.float().mean())
        out["outliers/y_sqnorm_share"] = float(y_sq[top_ids].sum() / y_sq.sum().clamp_min(1e-30))
        for low, high in POSITION_BINS:
            mask = position[top_ids] >= low
            if high is not None:
                mask &= position[top_ids] < high
            if high == low + 1:
                label = f"{low}"
            elif high is None:
                label = f"{low}+"
            else:
                label = f"{low}-{high - 1}"
            out[f"outliers/position_{label}"] = int(mask.sum())
        _, table_ids = self.top.top(cfg.outlier_table_size)
        out["outliers/top"] = [
            {
                "rank": rank + 1,
                "y_sqnorm": float(y_sq[i]),
                "x_sqnorm": float(x_sq[i]),
                "sample_error": float(error[i]),
                "position": int(position[i]),
                "sequence": int(meta["sequence"][i]),
                "shard": self.entries[int(meta["entry"][i])],
            }
            for rank, i in enumerate(table_ids.cpu().tolist())
        ]

        spectra: dict[str, torch.Tensor] = {}
        if full.eigenvalues is not None:
            spectra["full"] = full.eigenvalues
        if reference_full.eigenvalues is not None:
            spectra["reference"] = reference_full.eigenvalues  # same n: finite-sample spread
        subsets = {
            "excl_leading": (self.nonleading, self.reference_nonleading),
            "trimmed": (full.without(top_rows), reference_full.without(self.reference_top.top(k)[0])),
        }
        for name, (model_moments, reference_moments) in subsets.items():
            if model_moments.n < 2:
                continue
            out.update(model_moments.summary(f"gaussian/{name}/"))
            out.update(reference_moments.summary(f"reference/{name}/"))
            out[f"gaussian/{name}/count"] = model_moments.n
            spectra[name] = model_moments.eigenvalues
        return out, spectra


@torch.no_grad()
def evaluate_model(
    model: DenseSIGRegAE,
    source: DataSource,
    *,
    split: str,
    batch_size: int,
    maximum_batches: int,
    device: torch.device,
    amp_dtype: str,
    sigreg_cfg: SIGRegConfig,
    eval_cfg: EvalConfig,
    detailed: bool,
) -> dict[str, Any]:
    """Evaluate without touching any global or training random state."""
    was_training = model.training
    model.eval()
    d_in = model.cfg.d_in
    d_latent = model.cfg.d_latent
    t, weights = integration_grid(sigreg_cfg.t_min, sigreg_cfg.t_max, sigreg_cfg.num_points)
    validation_projections = fixed_projections(
        d_latent, sigreg_cfg.validation_projections, sigreg_cfg.validation_seed, "validation"
    ).to(device)
    diagnostic_projections = fixed_projections(
        d_latent, eval_cfg.diagnostic_projections, eval_cfg.diagnostic_seed, "diagnostic"
    ).to(device)
    reference_generator = torch.Generator().manual_seed(eval_cfg.gaussian_reference_seed)

    def projection_sets() -> dict[str, _ProjectionDiagnostics]:
        sets = {
            "heldout_": _ProjectionDiagnostics(
                validation_projections, t, weights, sigreg_cfg.scale_by_batch_size,
                None, 0, w2=False,
            )
        }
        if detailed:
            sets["diagnostic_"] = _ProjectionDiagnostics(
                diagnostic_projections, t, weights, sigreg_cfg.scale_by_batch_size,
                tuple(eval_cfg.quantiles), eval_cfg.maximum_quantile_samples, w2=True,
            )
        return sets

    model_sets = projection_sets()
    reference_sets = projection_sets()
    model_moments = _Moments(d_latent, device, covariance=detailed)
    reference_moments = _Moments(d_latent, device, covariance=detailed)
    n = 0
    sse = 0.0
    sse_normalized = 0.0
    sum_h = torch.zeros(d_in, dtype=torch.float64, device=device)
    sum_h_sq = 0.0
    batches = 0
    diagnostics = (
        _OutlierDiagnostics(d_latent, device, eval_cfg, source.paths(split)) if detailed else None
    )
    for item in eval_batches(
        source, split, batch_size, maximum_batches, seed=eval_cfg.sample_seed,
        with_metadata=detailed,
    ):
        batch, meta = item if detailed else (item, None)
        h = batch.to(device).float()
        with _autocast(device, amp_dtype):
            out = model(h)
        y = out["y"].float()
        x_hat = out["x_hat"].float()
        h_hat = model.denormalize(x_hat)
        n += len(h)
        sse += float((h_hat.double() - h.double()).square().sum())
        sse_normalized += float((x_hat.double() - out["x"].double()).square().sum())
        sum_h += h.double().sum(0)
        sum_h_sq += float(h.double().square().sum())
        gaussian = torch.randn(len(h), d_latent, generator=reference_generator).to(device)
        model_moments.add(y)
        reference_moments.add(gaussian)
        for key in model_sets:
            model_sets[key].add(y)
            reference_sets[key].add(gaussian)
        if diagnostics is not None:
            diagnostics.add(h, h_hat, out["x"].float(), x_hat, y, gaussian, meta)
        batches += 1
    if n == 0:
        raise ValueError(f"split {split!r} did not yield a single full batch of {batch_size}")
    mean_h = sum_h / n
    total_variation = sum_h_sq - n * float(mean_h.square().sum())
    results: dict[str, Any] = {
        "split": split,
        "positions": n,
        "batches": batches,
        "batch_size": batch_size,
        "detailed": detailed,
        "reconstruction/mse": sse / (n * d_in),
        "reconstruction/fvu": sse / max(total_variation, 1e-30),
        "reconstruction/normalized_mse": sse_normalized / (n * d_in),
        "sigreg_gaussian_expected_value": gaussian_expected_value(t, weights),
    }
    results.update(model_moments.summary("gaussian/"))
    results.update(reference_moments.summary("reference/"))
    for key in model_sets:
        results.update(model_sets[key].summary(f"gaussian/{key}"))
        results.update(reference_sets[key].summary(f"reference/{key}"))
    if diagnostics is not None:
        extra, spectra = diagnostics.summary(model_moments, reference_moments)
        results.update(extra)
        # Eigenvalue spectra are large; callers save them next to the JSON.
        results["_spectra"] = spectra
    if was_training:
        model.train()
    return results


def write_evaluation(results: dict[str, Any], output: Path) -> None:
    """Write metrics JSON and, when present, eigenvalue spectra as ``*-spectra.pt``."""
    spectra = results.pop("_spectra", None)
    if spectra:
        spectra_path = output.with_name(output.stem + "-spectra.pt")
        torch.save(spectra, spectra_path)
        results["spectra_path"] = spectra_path.name
    write_json(output, results)


def load_checkpoint_model(
    path: str | Path, device: torch.device
) -> tuple[DenseSIGRegAE, dict[str, Any]]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if state.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError(f"unsupported checkpoint {path}")
    cfg = config_from_dict(state["config"])
    model = build_model(
        cfg.model, state["normalization"]["mean"], state["normalization"]["scale"]
    )
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a stage-1 dense checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--output", help="JSON path (default: next to the run)")
    parser.add_argument("--batches", type=int, help="override eval.batches (0 = all)")
    parser.add_argument("--activation-manifest", help="override the stored manifest path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--set", action="append", default=[], metavar="eval.KEY=VALUE",
        help="override evaluation settings only, e.g. --set eval.leading_positions=4",
    )
    args = parser.parse_args()
    device = torch.device(
        args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    )
    model, state = load_checkpoint_model(args.checkpoint, device)
    cfg = config_from_dict(state["config"])
    for override in args.set:
        values = parse_override(override)
        if set(values) != {"eval"}:
            raise ValueError(f"only eval.* settings can be overridden here, got {override!r}")
        update_config(cfg, values)
    source = DataSource(
        args.activation_manifest or cfg.data.activation_manifest,
        skip_burn_in=cfg.data.skip_burn_in,
        test_split=cfg.data.test_split,
        holdout_test_fraction=cfg.data.holdout_test_fraction,
    )
    if source.fingerprint != state["data_manifest"]["fingerprint"]:
        raise ValueError("activation manifest differs from the one used in training")
    if source.splits != state["data_manifest"]["splits"]:
        raise ValueError("split assignment differs from the one used in training")
    results = evaluate_model(
        model,
        source,
        split=args.split,
        batch_size=cfg.eval.batch_size,
        maximum_batches=cfg.eval.batches if args.batches is None else args.batches,
        device=device,
        amp_dtype=cfg.optim.amp_dtype if device.type == "cuda" else "none",
        sigreg_cfg=cfg.sigreg,
        eval_cfg=cfg.eval,
        detailed=True,
    )
    results.update(
        {
            "checkpoint": str(args.checkpoint),
            "step": int(state["step"]),
            "sigreg_weight": cfg.sigreg.weight,
            "run_name": cfg.name,
            "seed": cfg.train.seed,
        }
    )
    output = (
        Path(args.output)
        if args.output
        else Path(args.checkpoint).parent.parent / f"eval-{args.split}-step-{int(state['step']):07d}.json"
    )
    write_evaluation(results, output)
    print(
        f"{args.split}: FVU={results['reconstruction/fvu']:.4f} "
        f"heldout SIGReg={results['gaussian/heldout_sigreg']:.3f} "
        f"(Gaussian ref {results['reference/heldout_sigreg']:.3f}); "
        f"top {results['outliers/count']} ||y||: "
        f"{results['outliers/leading_share']:.0%} at position < {results['diagnostic/leading_positions']}, "
        f"eff. rank full/excl-leading/trimmed = {results.get('gaussian/cov_effective_rank', float('nan')):.0f}/"
        f"{results.get('gaussian/excl_leading/cov_effective_rank', float('nan')):.0f}/"
        f"{results.get('gaussian/trimmed/cov_effective_rank', float('nan')):.0f}"
    )


if __name__ == "__main__":
    main()
