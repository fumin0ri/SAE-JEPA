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
      any SIGReg computation.
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

from .config import EvalConfig, SIGRegConfig, config_from_dict
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

    def add(self, y: torch.Tensor) -> None:
        y = y.double()
        self.n += len(y)
        self.sum += y.sum(0)
        self.sum_sq += y.square().sum(0)
        if self.second is not None:
            self.second += y.T @ y

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
    for batch in eval_batches(
        source, split, batch_size, maximum_batches, seed=eval_cfg.sample_seed
    ):
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
    if was_training:
        model.train()
    return results


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
    args = parser.parse_args()
    device = torch.device(
        args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    )
    model, state = load_checkpoint_model(args.checkpoint, device)
    cfg = config_from_dict(state["config"])
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
    write_json(output, results)
    print(
        f"{args.split}: FVU={results['reconstruction/fvu']:.4f} "
        f"heldout SIGReg={results['gaussian/heldout_sigreg']:.3f} "
        f"(Gaussian ref {results['reference/heldout_sigreg']:.3f})"
    )


if __name__ == "__main__":
    main()
