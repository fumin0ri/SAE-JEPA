import pytest
import torch

from sae_jepa.config import load_config
from sae_jepa.covariance import (gaussian_covariance_expected_value,
                                 sample_orthonormal_sketch, sketched_covariance_loss)
from sae_jepa.masked import MaskedTrainer
from test_masked import config


def test_sketch_is_orthonormal_reproducible_and_rng_isolated():
    rng = torch.Generator().manual_seed(13)
    state = rng.get_state()
    global_state = torch.get_rng_state()
    r = sample_orthonormal_sketch(32, 8, rng)
    torch.testing.assert_close(r.T @ r, torch.eye(8), atol=1e-6, rtol=1e-6)
    assert not torch.equal(r, sample_orthonormal_sketch(32, 8, rng))
    rng.set_state(state)
    assert torch.equal(r, sample_orthonormal_sketch(32, 8, rng))
    assert torch.equal(global_state, torch.get_rng_state())


def test_loss_matches_float64_covariance_and_is_translation_invariant():
    rng = torch.Generator().manual_seed(5)
    y = torch.randn(32, 12, generator=rng, requires_grad=True)
    r = sample_orthonormal_sketch(12, 4, rng)
    expected = (torch.cov((y.double() @ r.double()).T) - torch.eye(4)).square().mean()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = sketched_covariance_loss(y, r)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(actual, sketched_covariance_loss(y + 3, r))
    actual.backward()
    assert torch.isfinite(y.grad).all() and y.grad.norm() > 0


def test_gaussian_floor_and_anisotropy():
    rng = torch.Generator().manual_seed(44)
    samples = torch.randn(1000, 32, 4, generator=rng)
    centered = samples - samples.mean(1, keepdim=True)
    covariance = centered.transpose(1, 2) @ centered / 31
    observed = (covariance - torch.eye(4)).square().mean()
    assert float(observed) == pytest.approx(gaussian_covariance_expected_value(32, 4), rel=.05)
    y = samples.reshape(-1, 4)
    anisotropic = y * torch.tensor([2., 0., 0., 0.])  # same population trace
    assert sketched_covariance_loss(anisotropic, torch.eye(4)) > 100 * sketched_covariance_loss(y, torch.eye(4))


def test_split_half_matches_formula_and_is_translation_invariant():
    rng = torch.Generator().manual_seed(6)
    y = torch.randn(33, 12, generator=rng, requires_grad=True)
    r = sample_orthonormal_sketch(12, 4, rng)
    z = y.double() @ r.double()
    eye = torch.eye(4, dtype=torch.float64)
    expected = ((torch.cov(z[:16].T) - eye) * (torch.cov(z[16:].T) - eye)).mean()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = sketched_covariance_loss(y, r, "split_half")
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(actual, sketched_covariance_loss(y + 3, r, "split_half"))
    actual.backward()
    assert torch.isfinite(y.grad).all() and y.grad.norm() > 0
    with pytest.raises(ValueError, match="estimator"):
        sketched_covariance_loss(y, r, "bogus")


@pytest.mark.parametrize("scale", [1.0, 0.8])
def test_split_half_is_unbiased_while_plugin_prefers_shrinkage(scale):
    # Population covariance scale * I_k: the exact loss is (scale - 1)^2 / k.
    b, k = 32, 8
    rng = torch.Generator().manual_seed(45)
    samples = scale ** .5 * torch.randn(4000, b, k, generator=rng)
    split = torch.stack([sketched_covariance_loss(s, torch.eye(k), "split_half") for s in samples])
    plugin = torch.stack([sketched_covariance_loss(s, torch.eye(k)) for s in samples])
    exact = (scale - 1) ** 2 / k
    assert float(split.mean()) == pytest.approx(exact, abs=4 * float(split.std()) / len(split) ** .5)
    if scale == 1.0:
        assert gaussian_covariance_expected_value(b, k, "split_half") == 0.0
        # Shrinking toward (B-1)/(B+k) lowers the plug-in loss below its value at I.
        shrunk = ((b - 1) / (b + k)) ** .5 * samples
        assert float(torch.stack([sketched_covariance_loss(s, torch.eye(k)) for s in shrunk]).mean()) < float(plugin.mean())
    else:
        assert float(plugin.mean()) < gaussian_covariance_expected_value(b, k)


def test_split_half_corr_matches_formula_and_ignores_scale():
    rng = torch.Generator().manual_seed(7)
    y = torch.randn(33, 12, generator=rng) @ torch.randn(12, 12, generator=rng)
    y.requires_grad_(True)
    r = sample_orthonormal_sketch(12, 4, rng)
    z = y.detach().double() @ r.double()
    off = 1 - torch.eye(4, dtype=torch.float64)
    expected = (torch.corrcoef(z[:16].T) * torch.corrcoef(z[16:].T) * off).sum() / 16
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = sketched_covariance_loss(y, r, "split_half_corr")
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=1e-4, atol=1e-6)
    # Invariant to per-direction scale in the sketch, hence no pull on variance.
    scales = torch.tensor([.1, 1., 3., 10.])
    torch.testing.assert_close(
        sketched_covariance_loss(y.detach() @ r * scales, torch.eye(4), "split_half_corr"),
        sketched_covariance_loss(y.detach() @ r, torch.eye(4), "split_half_corr"), rtol=1e-4, atol=1e-6)
    actual.backward()
    assert torch.isfinite(y.grad).all() and y.grad.norm() > 0
    assert float((y.grad * y.detach()).sum()) == pytest.approx(0, abs=1e-4 * float(y.grad.norm() * y.detach().norm()))


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_split_half_corr_estimates_squared_correlation(rho):
    # Non-Gaussian marginals: zero mean for independent coordinates; for rho != 0
    # sample correlations shrink toward 0 by O(1/n) (about 3% here at n=32).
    b, k = 64, 2
    rng = torch.Generator().manual_seed(46)
    u = torch.rand(4000, b, k, generator=rng) - .5
    samples = torch.stack([u[..., 0], rho * u[..., 0] + (1 - rho ** 2) ** .5 * u[..., 1]], -1) * 5
    losses = torch.stack([sketched_covariance_loss(s, torch.eye(k), "split_half_corr") for s in samples])
    exact = 2 * rho ** 2 / k ** 2
    tolerance = 4 * float(losses.std()) / len(losses) ** .5
    if rho == 0:
        assert float(losses.mean()) == pytest.approx(0, abs=tolerance)
    else:
        assert .94 * exact < float(losses.mean()) < exact + tolerance
    assert gaussian_covariance_expected_value(b, k, "split_half_corr") == 0.0


@pytest.mark.parametrize("override", ["covariance.weight=-1", "covariance.weight=.nan",
    "covariance.weight=.inf", "covariance.sketch_dim=0", "covariance.sketch_dim=1.5",
    "covariance.sketch_dim=512", "covariance.sketch_dim=4097", "covariance.estimator=bogus"])
def test_invalid_settings(override):
    with pytest.raises(ValueError, match="covariance"):
        load_config(overrides=["model.type=masked_sigreg_encoder", "sigreg.weight=.1",
                               "covariance.weight=1", override])


@pytest.mark.parametrize("masked,estimator", [(True, "plugin"), (False, "plugin"),
                                              (True, "split_half"), (False, "split_half_corr")])
def test_objective_gradients_evaluation_and_exact_resume(manifest, tmp_path, masked, estimator):
    def make(path):
        cfg = config(manifest, path)
        cfg.masking.enabled = masked
        cfg.covariance.weight = 2.0
        cfg.covariance.sketch_dim = 4
        cfg.covariance.estimator = estimator
        return MaskedTrainer(cfg)

    trainer = make(tmp_path / "straight")
    loss, metrics = trainer.loss(next(trainer.data), True)
    assert float(loss.detach()) == pytest.approx(
        metrics.get("consistency_mse", 0) + metrics["sigreg_weighted"] + metrics["covariance_weighted"])
    views = ("full", "masked") if masked else ("full",)
    assert metrics["covariance"] == pytest.approx(sum(metrics[f"covariance_{v}"] for v in views) / len(views))
    assert all(metrics[f"grad_rms_y/{v}/covariance_weighted"] > 0 for v in views)
    state = trainer.covariance_generator.get_state()
    first, second = trainer.validate(), trainer.validate()
    assert first == second
    assert "reference/sketched_covariance" in first
    assert first["covariance_estimator"] == estimator
    assert trainer.convention["covariance"]["estimator"] == estimator
    assert torch.equal(state, trainer.covariance_generator.get_state())

    # Start both runs afresh, then compare uninterrupted and resumed training.
    trainer = make(tmp_path / "complete")
    trainer.run()
    partial = make(tmp_path / "resume")
    partial.run(max_steps=4)
    resumed = make(tmp_path / "resume")
    checkpoint = tmp_path / "resume/checkpoints/latest.pt"
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key
    assert torch.equal(trainer.covariance_generator.get_state(), resumed.covariance_generator.get_state())
    resumed.cfg.covariance.weight = 3
    with pytest.raises(ValueError, match="covariance.weight"):
        MaskedTrainer(resumed.cfg).load_checkpoint(checkpoint)
    resumed.cfg.covariance.weight = 2.0
    resumed.cfg.covariance.estimator = "split_half" if estimator == "plugin" else "plugin"
    with pytest.raises(ValueError, match="covariance.estimator"):
        MaskedTrainer(resumed.cfg).load_checkpoint(checkpoint)


def test_disabled_skips_sketch_and_old_checkpoint_remains_resumable(manifest, tmp_path, monkeypatch):
    import sae_jepa.masked as module
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled covariance must not draw sketches")
    monkeypatch.setattr(module, "sample_orthonormal_sketch", forbidden)
    cfg = config(manifest, tmp_path / "old")
    trainer = MaskedTrainer(cfg)
    trainer.loss(next(trainer.data), True)
    trainer.validate()
    state = trainer.checkpoint_state()
    del state["config"]["covariance"]
    assert "covariance" not in state["rng"]
    path = tmp_path / "legacy.pt"
    torch.save(state, path)
    restored = MaskedTrainer(cfg)
    restored.load_checkpoint(path)
    assert restored.covariance_generator is None
