from __future__ import annotations

import json

import pytest
import torch

import sae_jepa.sigreg as sigreg_module
from sae_jepa.evaluate import load_checkpoint_model
from sae_jepa.frontends import load_frontend
from sae_jepa.reporting import collect, write_report
from sae_jepa.train import Trainer, lr_multiplier

from conftest import tiny_config


def test_lr_schedule_warmup_constant_decay_to_zero():
    values = [lr_multiplier(i, 100, 10, 0.2) for i in range(100)]
    assert values[0] == pytest.approx(0.1) and values[9] == pytest.approx(1.0)
    assert all(v == 1.0 for v in values[10:80])
    assert values[80] == pytest.approx(1.0) and values[99] == pytest.approx(0.05)
    assert all(a >= b for a, b in zip(values[80:], values[81:]))
    assert lr_multiplier(100, 100, 10, 0.2) == 0.0


def test_both_losses_send_finite_gradients_to_encoder(manifest, tmp_path):
    trainer = Trainer(tiny_config(manifest, tmp_path / "run"))
    h = next(trainer.data)
    loss, metrics = trainer.loss(h, diagnostics=True)
    assert metrics["grad_rms_y/reconstruction"] > 0
    assert metrics["grad_rms_y/sigreg_weighted"] > 0
    terms = (
        lambda out: torch.nn.functional.mse_loss(out["x_hat"], out["x"]),
        lambda out: trainer.sigreg(out["y"]),
    )
    for term in terms:
        trainer.model.zero_grad()
        term(trainer.model(h)).backward()
        grads = [p.grad for p in trainer.model.encoder.parameters()]
        assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_zero_weight_skips_sigreg_and_its_rng(manifest, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("SIGReg must not run in the training step when weight == 0")

    trainer = Trainer(tiny_config(manifest, tmp_path / "run", weight=0.0))
    assert trainer.sigreg is None
    monkeypatch.setattr(sigreg_module, "sample_projections", forbidden)
    for diagnostics in (True, False):
        metrics = trainer.train_step(diagnostics)
    assert "sigreg" not in metrics


def test_conditions_share_initialization_and_data_order(manifest, tmp_path):
    a = Trainer(tiny_config(manifest, tmp_path / "a", weight=0.0))
    b = Trainer(tiny_config(manifest, tmp_path / "b", weight=1.0))
    for (name, p), (_, q) in zip(a.model.state_dict().items(), b.model.state_dict().items()):
        assert torch.equal(p, q), name
    for _ in range(5):
        assert torch.equal(next(a.data), next(b.data))


def test_resume_reproduces_uninterrupted_run(manifest, tmp_path):
    straight = Trainer(tiny_config(manifest, tmp_path / "straight"))
    straight.run()

    interrupted = Trainer(tiny_config(manifest, tmp_path / "resumed"))
    interrupted.run(max_steps=4)
    assert (tmp_path / "resumed" / "checkpoints" / "latest.pt").exists()
    resumed = Trainer(tiny_config(manifest, tmp_path / "resumed"))
    resumed.load_checkpoint(tmp_path / "resumed" / "checkpoints" / "latest.pt")
    assert resumed.step == 4
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(
        interrupted.optimizer.param_groups[0]["lr"]
    )
    resumed.run()
    for key, value in straight.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key
    assert straight.sigreg.draws == resumed.sigreg.draws == 8


def test_validation_does_not_touch_training_randomness(manifest, tmp_path):
    trainer = Trainer(tiny_config(manifest, tmp_path / "run"))
    torch_state = torch.get_rng_state()
    sigreg_state = trainer.sigreg.generator.get_state()
    data_state = trainer.data.state_dict()
    first = trainer.validate(detailed=True)
    second = trainer.validate(detailed=True)
    assert first == second
    assert torch.equal(torch_state, torch.get_rng_state())
    assert torch.equal(sigreg_state, trainer.sigreg.generator.get_state())
    assert data_state == trainer.data.state_dict()
    for key in ("reconstruction/fvu", "gaussian/heldout_sigreg", "reference/heldout_sigreg",
                "gaussian/diagnostic_w2_sq", "reference/diagnostic_w2_sq", "gaussian/cov_fro_dev"):
        assert key in first and first[key] == first[key]


def test_end_to_end_checkpoint_frontend_and_report(manifest, tmp_path):
    run = tmp_path / "runs" / "lambda-0p1"
    final = Trainer(tiny_config(manifest, run)).run()
    assert final["step"] == 8
    lines = (run / "metrics.jsonl").read_text().splitlines()
    assert any("validation" in json.loads(line) for line in lines)
    checkpoint = run / "checkpoints" / "latest.pt"
    state = torch.load(checkpoint, weights_only=False)
    for key in ("model", "optimizer", "scheduler", "rng", "normalization", "data_manifest",
                "config", "sigreg_convention"):
        assert key in state
    model, _ = load_checkpoint_model(checkpoint, torch.device("cpu"))
    frontend = load_frontend("dense_sigreg", checkpoint)
    h = torch.randn(4, 16)
    assert torch.equal(frontend.encode_dense(h), model.encode_dense(h))
    assert not any(p.requires_grad for p in frontend.parameters())
    rows = collect(tmp_path / "runs")
    write_report(rows, tmp_path / "report")
    assert (tmp_path / "report" / "validation.md").exists()
