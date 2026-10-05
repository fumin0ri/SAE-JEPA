"""Random orthonormal sketches of the centered, unbiased sample covariance.

Estimators of second-moment losses on z = y R, none with a batch-size factor.
They constrain second moments, not the output mean.

* ``plugin``: the squared error of the full-batch sample covariance. Its
  expectation carries a finite-batch noise floor, and for a population
  covariance ``a I_k`` it is minimized at ``a = (B-1)/(B+k)``, not 1.
* ``split_half``: ``mean((C_1 - I) * (C_2 - I))`` with C_1, C_2 the sample
  covariances of the two batch halves, each centered on its own mean. For
  independent halves its expectation is exactly the population loss, so it is
  minimized at Cov = I and is zero in expectation for N(0, I). It can be
  negative. When correlations remain, ||s^2 R - I||^2 is still minimized at
  s^2 = k / ||R||_F^2 < 1, so this loss also pulls the scale down.
* ``split_half_corr``: ``sum_{i != j} r1_ij * r2_ij / k^2`` with r1, r2 the
  sample correlation matrices of the two halves. It estimates the squared
  off-diagonal population correlations, leaves every per-direction scale to
  SIGReg, and so exerts no pull on variance. It is zero in expectation for
  any distribution with independent coordinates and can be negative; nonzero
  correlations are underestimated by O(1/n) per half.
"""
from __future__ import annotations

import torch


def sample_orthonormal_sketch(d: int, k: int, generator: torch.Generator,
                              device: torch.device | str = "cpu") -> torch.Tensor:
    """Draw on a dedicated CPU RNG, then reduced QR on the target device."""
    if not 1 <= k <= d:
        raise ValueError("sketch_dim must satisfy 1 <= k <= d")
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        raw = torch.randn(d, k, generator=generator, dtype=torch.float32).to(device)
        q, r = torch.linalg.qr(raw, mode="reduced")
        # Fix QR's sign convention to obtain a uniform orthonormal frame.
        return q * torch.where(r.diagonal() < 0, -1.0, 1.0)


ESTIMATORS = ("plugin", "split_half", "split_half_corr")
CORRELATION_EPS = 1e-8  # added to the variance before the square root


def _centered_covariance(z: torch.Tensor) -> torch.Tensor:
    z = z - z.mean(0, keepdim=True)
    return z.T @ z / (len(z) - 1)


def _off_diagonal_correlation(z: torch.Tensor) -> torch.Tensor:
    covariance = _centered_covariance(z)
    inverse_std = (covariance.diagonal() + CORRELATION_EPS).rsqrt()
    correlation = covariance * inverse_std[:, None] * inverse_std[None, :]
    return correlation - torch.diag(correlation.diagonal())


def sketched_covariance_loss(y: torch.Tensor, projection: torch.Tensor,
                             estimator: str = "plugin") -> torch.Tensor:
    """Float32 covariance loss; caller supplies an orthonormal [d, k] sketch.

    The split-half estimators use the first B//2 rows and the rest.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown covariance estimator {estimator!r}")
    if y.ndim != 2 or projection.ndim != 2 or y.shape[1] != projection.shape[0]:
        raise ValueError("expected y [B, d] and projection [d, k]")
    b, d = y.shape
    k = projection.shape[1]
    if not 1 <= k <= d or b <= k:
        raise ValueError("require 1 <= sketch_dim <= d and sketch_dim < batch size")
    with torch.autocast(device_type=y.device.type, enabled=False):
        z = y.float() @ projection.to(device=y.device, dtype=torch.float32)
        identity = torch.eye(k, device=y.device, dtype=torch.float32)
        if estimator == "plugin":
            return (_centered_covariance(z) - identity).square().mean()
        half = b // 2
        if estimator == "split_half_corr":
            first = _off_diagonal_correlation(z[:half])
            second = _off_diagonal_correlation(z[half:])
        else:
            first = _centered_covariance(z[:half]) - identity
            second = _centered_covariance(z[half:]) - identity
        return (first * second).mean()


def gaussian_covariance_expected_value(batch_size: int, k: int,
                                       estimator: str = "plugin") -> float:
    """Expected loss for iid N(0,I), independent of an orthonormal sketch."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown covariance estimator {estimator!r}")
    if not 1 <= k < batch_size:
        raise ValueError("require 1 <= sketch_dim < batch size")
    if estimator != "plugin":
        return 0.0
    return (k + 1) / (k * (batch_size - 1))
