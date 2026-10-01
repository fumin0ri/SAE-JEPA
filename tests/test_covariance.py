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


@pytest.mark.parametrize("override", ["covariance.weight=-1", "covariance.weight=.nan",
    "covariance.weight=.inf", "covariance.sketch_dim=0", "covariance.sketch_dim=1.5",
    "covariance.sketch_dim=512", "covariance.sketch_dim=4097"])
def test_invalid_settings(override):
    with pytest.raises(ValueError, match="covariance"):
        load_config(overrides=["model.type=masked_sigreg_encoder", "sigreg.weight=.1",
                               "covariance.weight=1", override])


@pytest.mark.parametrize("masked", [True, False])
def test_objective_gradients_evaluation_and_exact_resume(manifest, tmp_path, masked):
    def make(path):
        cfg = config(manifest, path)
        cfg.masking.enabled = masked
        cfg.covariance.weight = 2.0
        cfg.covariance.sketch_dim = 4
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
