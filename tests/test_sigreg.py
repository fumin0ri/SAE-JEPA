from __future__ import annotations

import math

import pytest
import torch

from sae_jepa.sigreg import (
    SIGRegLoss,
    epps_pulley_sigreg,
    gaussian_expected_value,
    integration_grid,
    reference_epps_pulley_sigreg,
    sample_projections,
)


def _setup(d=16, m=12, seed=0):
    generator = torch.Generator().manual_seed(seed)
    t, w = integration_grid(-5.0, 5.0, 17)
    return generator, sample_projections(d, m, generator), t, w


def test_grid_is_symmetric_trapezoid():
    t, w = integration_grid(-5.0, 5.0, 17)
    assert torch.allclose(t, -t.flip(0))
    assert math.isclose(float(w.sum()), 10.0, rel_tol=1e-6)
    assert math.isclose(float(w[0]), 0.3125) and math.isclose(float(w[1]), 0.625)


def test_projections_are_unit_norm():
    _, projections, _, _ = _setup(d=32, m=64)
    assert torch.allclose(projections.norm(dim=0), torch.ones(64), atol=1e-5)


@pytest.mark.parametrize("scaled", [True, False])
def test_matches_literal_reference(scaled):
    generator, projections, t, w = _setup()
    y = torch.randn(40, 16, generator=generator) * 1.3 + 0.2
    fast = float(epps_pulley_sigreg(y, projections, t, w, scaled))
    slow = reference_epps_pulley_sigreg(y, projections, t, w, scaled)
    assert fast == pytest.approx(slow, rel=1e-4, abs=1e-6)


def test_gaussian_is_near_noise_floor_and_distinguishes_departures():
    generator = torch.Generator().manual_seed(1)
    d, n = 32, 512
    t, w = integration_grid()
    projections = sample_projections(d, 256, generator)
    expected = gaussian_expected_value(t, w)
    assert expected == pytest.approx(math.sqrt(2 * math.pi) - math.sqrt(2 * math.pi / 3), rel=2e-2)

    def value(y):
        return float(epps_pulley_sigreg(y, projections, t, w))

    gaussian = value(torch.randn(n, d, generator=generator))
    assert gaussian == pytest.approx(expected, rel=0.25)
    shifted = value(torch.randn(n, d, generator=generator) + 0.5)
    wide = value(2.0 * torch.randn(n, d, generator=generator))
    narrow = value(0.5 * torch.randn(n, d, generator=generator))
    collapsed = value(torch.zeros(n, d))
    rank_one = value(torch.randn(n, 1, generator=generator).repeat(1, d) / math.sqrt(d) * 3)
    for bad in (shifted, wide, narrow, collapsed, rank_one):
        assert bad > 5 * gaussian


def test_gradient_is_finite_and_moves_towards_gaussian():
    generator, projections, t, w = _setup(d=8, m=16)
    y = (0.2 * torch.randn(128, 8, generator=generator) + 1.0).requires_grad_()
    loss = epps_pulley_sigreg(y, projections, t, w)
    loss.backward()
    assert torch.isfinite(y.grad).all() and y.grad.abs().sum() > 0
    with torch.no_grad():
        stepped = y - 0.05 * y.grad / y.grad.norm() * y.norm()
    assert float(epps_pulley_sigreg(stepped, projections, t, w)) < float(loss.detach())


def test_loss_object_uses_private_generator_and_resamples():
    torch.manual_seed(0)
    before = torch.get_rng_state()
    loss = SIGRegLoss(8, num_projections=4, t_min=-5, t_max=5, num_points=17,
                      scale_by_batch_size=True, resample_every_step=True, seed=3)
    a, b = loss.projections(), loss.projections()
    assert not torch.allclose(a, b)
    assert torch.equal(before, torch.get_rng_state())
    state = loss.state_dict()
    c = loss.projections()
    loss.load_state_dict(state)
    assert torch.equal(c, loss.projections())
