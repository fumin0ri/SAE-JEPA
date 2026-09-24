"""Epps--Pulley SIGReg (LeJEPA) towards an isotropic standard Gaussian.

For unit projection directions ``a_m`` and projected samples
``u_im = a_m^T y_i`` the loss is

    L = (N / M) * sum_m  int | (1/N) sum_i exp(i t u_im) - exp(-t^2/2) |^2
                             * exp(-t^2/2) dt

The integral is a trapezoid rule on a symmetric grid (default: 17 points on
[-5, 5]).  The empirical characteristic function is computed from cos / sin in
float32 regardless of autocast.  For ``y ~ N(0, I)`` the N-scaled statistic
has expectation close to ``sqrt(2 pi) - sqrt(2 pi / 3) ~= 1.06`` per
projection, so values near 1 are the noise floor, not an imperfect fit.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .data import mix_seed


def integration_grid(
    t_min: float = -5.0, t_max: float = 5.0, num_points: int = 17
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodes and trapezoid weights on ``[t_min, t_max]``."""
    if num_points < 2 or not t_max > t_min:
        raise ValueError("need at least two points on a non-empty interval")
    t = torch.linspace(t_min, t_max, num_points, dtype=torch.float32)
    step = (t_max - t_min) / (num_points - 1)
    weights = torch.full((num_points,), step, dtype=torch.float32)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return t, weights


def sample_projections(
    d: int, num_projections: int, generator: torch.Generator
) -> torch.Tensor:
    """``[d, M]`` matrix of unit-norm columns, sampled on CPU for reproducibility."""
    directions = torch.randn(d, num_projections, generator=generator, dtype=torch.float32)
    return directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)


def epps_pulley_per_projection(
    y: torch.Tensor,
    projections: torch.Tensor,
    t: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Un-scaled integral for each projection, shape ``[M]`` (float32)."""
    if y.ndim != 2:
        raise ValueError("y must have shape [batch, d]")
    device = y.device
    with torch.autocast(device_type=device.type, enabled=False):
        u = y.float() @ projections.to(device=device, dtype=torch.float32)
        t = t.to(device=device, dtype=torch.float32)
        weights = weights.to(device=device, dtype=torch.float32)
        tu = u.unsqueeze(-1) * t  # [N, M, T]
        real = torch.cos(tu).mean(dim=0)
        imaginary = torch.sin(tu).mean(dim=0)
        target = torch.exp(-0.5 * t.square())
        error = (real - target).square() + imaginary.square()
        return (error * target * weights).sum(dim=-1)


def epps_pulley_sigreg(
    y: torch.Tensor,
    projections: torch.Tensor,
    t: torch.Tensor,
    weights: torch.Tensor,
    scale_by_batch_size: bool = True,
) -> torch.Tensor:
    per_projection = epps_pulley_per_projection(y, projections, t, weights)
    value = per_projection.mean()
    return value * y.shape[0] if scale_by_batch_size else value


def reference_epps_pulley_sigreg(
    y: torch.Tensor,
    projections: torch.Tensor,
    t: torch.Tensor,
    weights: torch.Tensor,
    scale_by_batch_size: bool = True,
) -> float:
    """Literal float64 / complex evaluation of the formula (tests only)."""
    y = y.double()
    n = y.shape[0]
    total = 0.0
    for m in range(projections.shape[1]):
        u = y @ projections[:, m].double()
        integral = 0.0
        for t_j, w_j in zip(t.double().tolist(), weights.double().tolist()):
            ecf = torch.exp(1j * t_j * u.to(torch.complex128)).mean()
            gaussian = math.exp(-0.5 * t_j * t_j)
            integral += w_j * abs(complex(ecf) - gaussian) ** 2 * gaussian
        total += integral
    value = total / projections.shape[1]
    return value * n if scale_by_batch_size else value


def gaussian_expected_value(t: torch.Tensor, weights: torch.Tensor) -> float:
    """Exact E[L] per projection for i.i.d. N(0,1) samples under the quadrature.

    E|ecf(t) - phi(t)|^2 = (1 - phi(t)^2) / N, so the N-scaled statistic is
    ``sum_j w_j (1 - phi_j^2) phi_j`` independent of N.
    """
    phi = torch.exp(-0.5 * t.double().square())
    return float((weights.double() * (1.0 - phi.square()) * phi).sum())


class SIGRegLoss:
    """Train-time SIGReg with its own generator, separate from all other RNG."""

    def __init__(
        self,
        d: int,
        *,
        num_projections: int,
        t_min: float,
        t_max: float,
        num_points: int,
        scale_by_batch_size: bool,
        resample_every_step: bool,
        seed: int,
    ):
        self.d = d
        self.num_projections = num_projections
        self.scale_by_batch_size = scale_by_batch_size
        self.resample_every_step = resample_every_step
        self.t, self.weights = integration_grid(t_min, t_max, num_points)
        self.generator = torch.Generator().manual_seed(seed)
        self._fixed: torch.Tensor | None = None
        self.draws = 0

    def projections(self) -> torch.Tensor:
        if not self.resample_every_step:
            if self._fixed is None:
                self._fixed = sample_projections(self.d, self.num_projections, self.generator)
                self.draws += 1
            return self._fixed
        self.draws += 1
        return sample_projections(self.d, self.num_projections, self.generator)

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        return epps_pulley_sigreg(
            y, self.projections(), self.t, self.weights, self.scale_by_batch_size
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "generator": self.generator.get_state(),
            "draws": self.draws,
            "fixed": self._fixed,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.generator.set_state(state["generator"])
        self.draws = int(state["draws"])
        self._fixed = state.get("fixed")


def fixed_projections(d: int, num_projections: int, seed: int, purpose: str) -> torch.Tensor:
    """Held-out projections from a dedicated generator (never the train one)."""
    tag = sum(ord(c) * (i + 1) for i, c in enumerate(purpose))
    generator = torch.Generator().manual_seed(mix_seed(seed, tag))
    return sample_projections(d, num_projections, generator)


def loss_convention(
    *, num_projections: int, t_min: float, t_max: float, num_points: int,
    scale_by_batch_size: bool, resample_every_step: bool, batch_size: int,
) -> dict[str, Any]:
    t, weights = integration_grid(t_min, t_max, num_points)
    return {
        "name": "Epps-Pulley SIGReg (LeJEPA)",
        "formula": "(N/M) sum_m int |ecf_m(t) - exp(-t^2/2)|^2 exp(-t^2/2) dt"
        if scale_by_batch_size
        else "(1/M) sum_m int |ecf_m(t) - exp(-t^2/2)|^2 exp(-t^2/2) dt",
        "target": "isotropic standard Gaussian N(0, I)",
        "projections": num_projections,
        "projection_distribution": "uniform on the unit sphere (normalized Gaussian)",
        "resample_every_step": resample_every_step,
        "quadrature": f"trapezoid, {num_points} points on [{t_min}, {t_max}]",
        "batch_size_factor_N": scale_by_batch_size,
        "batch_size": batch_size,
        "precision": "float32 cos/sin, autocast disabled",
        "gaussian_expected_value_per_projection": gaussian_expected_value(t, weights),
    }
