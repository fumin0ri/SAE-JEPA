"""Random orthonormal sketches of the centered, unbiased sample covariance.

The loss is ||Cov(y R) - I_k||_F^2 / k^2, with no batch-size factor or
noise-floor subtraction. It constrains second moments, not the output mean.
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


def sketched_covariance_loss(y: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """Float32 covariance loss; caller supplies an orthonormal [d, k] sketch."""
    if y.ndim != 2 or projection.ndim != 2 or y.shape[1] != projection.shape[0]:
        raise ValueError("expected y [B, d] and projection [d, k]")
    b, d = y.shape
    k = projection.shape[1]
    if not 1 <= k <= d or b <= k:
        raise ValueError("require 1 <= sketch_dim <= d and sketch_dim < batch size")
    with torch.autocast(device_type=y.device.type, enabled=False):
        z = y.float() @ projection.to(device=y.device, dtype=torch.float32)
        z = z - z.mean(0, keepdim=True)
        covariance = z.T @ z / (b - 1)
        identity = torch.eye(k, device=y.device, dtype=torch.float32)
        return (covariance - identity).square().mean()


def gaussian_covariance_expected_value(batch_size: int, k: int) -> float:
    """Expected loss for iid N(0,I), independent of an orthonormal sketch."""
    if not 1 <= k < batch_size:
        raise ValueError("require 1 <= sketch_dim < batch size")
    return (k + 1) / (k * (batch_size - 1))
