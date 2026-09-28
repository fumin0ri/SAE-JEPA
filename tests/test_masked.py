import json

import pytest
import torch

from conftest import tiny_config
from sae_jepa.config import load_config
from sae_jepa.evaluate import load_checkpoint_model
from sae_jepa.masked import MaskedTrainer, main, mask_coordinates
from sae_jepa.sigreg import epps_pulley_sigreg
from sae_jepa.train import Trainer


def config(manifest, path):
    cfg = tiny_config(manifest, path)
    cfg.model.type = "masked_sigreg_encoder"
    cfg.masking.probability = 0.5
    return cfg


def test_masks_are_independent_zero_replacements_without_scaling():
    x = torch.full((1000, 16), 3.0)
    rng = torch.Generator().manual_seed(8)
    global_state = torch.get_rng_state()
    masked = mask_coordinates(x, .25, rng)
    assert set(masked.unique().tolist()) == {0., 3.}
    assert float((masked == 0).float().mean()) == pytest.approx(.25, abs=.02)
    assert not torch.equal(masked[0], masked[1])
    assert not torch.equal(masked, mask_coordinates(x, .25, rng))
    assert torch.equal(global_state, torch.get_rng_state())


def test_objective_matches_two_views_and_gradients_flow_both_ways(manifest, tmp_path):
    trainer = MaskedTrainer(config(manifest, tmp_path / "run"))
    h = next(trainer.data)
    mg = torch.Generator(); mg.set_state(trainer.mask_generator.get_state())
    projection_state = trainer.sigreg.state_dict()
    x = trainer.model.normalize(h)
    full = trainer.model.encode_normalized(x)
    masked = trainer.model.encode_normalized(mask_coordinates(x, .5, mg))
    projections = trainer.sigreg.projections()
    terms = [epps_pulley_sigreg(v, projections, trainer.sigreg.t, trainer.sigreg.weights)
             for v in (full, masked)]
    expected = (full - masked).square().mean() + trainer.cfg.sigreg.weight * sum(terms) / 2
    trainer.sigreg.load_state_dict(projection_state)
    actual, metrics = trainer.loss(h, True)
    torch.testing.assert_close(actual, expected)
    assert trainer.sigreg.draws == 1
    assert all(metrics[f"grad_rms_y/{view}/{term}"] > 0
               for view in ("full", "masked") for term in ("consistency", "sigreg_weighted"))
    actual.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainer.model.parameters())
    assert not any("decoder" in key for key in trainer.model.state_dict())


def test_resume_reproduces_masks_projections_and_parameters(manifest, tmp_path):
    straight = MaskedTrainer(config(manifest, tmp_path / "straight"))
    straight.run()
    interrupted = MaskedTrainer(config(manifest, tmp_path / "resume"))
    interrupted.run(max_steps=4)
    resumed = MaskedTrainer(config(manifest, tmp_path / "resume"))
    resumed.load_checkpoint(tmp_path / "resume/checkpoints/latest.pt")
    resumed.run()
    for key, value in straight.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key
    assert torch.equal(straight.mask_generator.get_state(), resumed.mask_generator.get_state())
    assert torch.equal(straight.sigreg.generator.get_state(), resumed.sigreg.generator.get_state())


def test_evaluation_is_repeatable_and_does_not_consume_training_rng(manifest, tmp_path):
    trainer = MaskedTrainer(config(manifest, tmp_path / "run"))
    states = [torch.get_rng_state(), trainer.mask_generator.get_state(), trainer.sigreg.generator.get_state()]
    data_state = trainer.data.state_dict()
    a, b = trainer.validate(True), trainer.validate(True)
    sa, sb = a.pop("_spectra"), b.pop("_spectra")
    assert a == b
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert not any(k.startswith("reconstruction/") for k in a)
    assert all(f"{view}/cov_effective_rank" in a for view in ("gaussian", "masked", "reference"))
    assert trainer.model.training
    assert data_state == trainer.data.state_dict()
    for before, after in zip(states, [torch.get_rng_state(), trainer.mask_generator.get_state(), trainer.sigreg.generator.get_state()]):
        assert torch.equal(before, after)


def test_offline_evaluate_and_report_cli(manifest, tmp_path):
    cfg = config(manifest, tmp_path / "run")
    cfg.data.skip_leading_positions = 1
    trained = MaskedTrainer(cfg).run()
    path = tmp_path / "run/checkpoints/latest.pt"
    output = tmp_path / "run/offline.json"
    main(["evaluate", "--checkpoint", str(path), "--device", "cpu", "--output", str(output)])
    offline = json.loads(output.read_text())
    assert offline["consistency/mse"] == trained["consistency/mse"]
    assert offline["gaussian/cov_effective_rank"] == trained["gaussian/cov_effective_rank"]
    main(["report", "--run-root", str(tmp_path)])
    assert "masked" in (tmp_path / "report/validation.md").read_text()
    assert (tmp_path / "report/validation_spectra.png").exists()
    with pytest.raises(ValueError, match="no reconstruction decoder"):
        load_checkpoint_model(path, torch.device("cpu"))


def test_mask_change_cannot_resume(manifest, tmp_path):
    cfg = config(manifest, tmp_path / "run")
    trainer = MaskedTrainer(cfg)
    path = trainer.save_checkpoint()
    cfg.masking.probability = .25
    with pytest.raises(ValueError, match="masking.probability"):
        MaskedTrainer(cfg).load_checkpoint(path)


@pytest.mark.parametrize("probability", [0, 1, -.1, float("nan")])
def test_invalid_mask_probabilities(probability):
    with pytest.raises(ValueError, match="masking.probability"):
        load_config(overrides=[f"masking.probability={probability}"])


def test_masked_config_requires_sigreg_and_dedicated_entrypoint(manifest, tmp_path):
    cfg = config(manifest, tmp_path / "run")
    with pytest.raises(ValueError, match="sj-masked train"):
        Trainer(cfg)
    cfg.sigreg.weight = 0
    with pytest.raises(ValueError, match="sigreg.weight"):
        MaskedTrainer(cfg)


def test_sigreg_only_has_no_mask_or_consistency_and_resumes(manifest, tmp_path, monkeypatch):
    import sae_jepa.masked as module

    def forbidden(*args, **kwargs):
        raise AssertionError("SIGReg-only must not generate masked views")

    monkeypatch.setattr(module, "mask_coordinates", forbidden)
    cfg = config(manifest, tmp_path / "only")
    cfg.masking.enabled = False
    cfg.sigreg.weight = 1.0
    trainer = MaskedTrainer(cfg)
    h = next(trainer.data)
    state = trainer.sigreg.state_dict()
    expected = trainer.sigreg(trainer.model.encode_dense(h))
    trainer.sigreg.load_state_dict(state)
    mask_state = trainer.mask_generator.get_state()
    loss, metrics = trainer.loss(h, True)
    torch.testing.assert_close(loss, expected)
    assert not any("consistency" in k or "masked" in k for k in metrics)
    assert torch.equal(mask_state, trainer.mask_generator.get_state())
    trainer.run(max_steps=4)
    resumed = MaskedTrainer(cfg)
    checkpoint = tmp_path / "only/checkpoints/latest.pt"
    resumed.load_checkpoint(checkpoint)
    trainer.run()
    result = resumed.run()
    assert result["objective"] == "sigreg_only"
    assert not any("consistency" in k or k.startswith("masked/") for k in result)
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key])
    output = tmp_path / "only/offline.json"
    main(["evaluate", "--checkpoint", str(checkpoint), "--device", "cpu", "--output", str(output)])
    assert json.loads(output.read_text())["gaussian/heldout_sigreg"] == result["gaussian/heldout_sigreg"]
    main(["report", "--run-root", str(tmp_path / "only")])
    assert "sigreg_only" in (tmp_path / "only/report/validation.md").read_text()
    cfg.masking.enabled = True
    with pytest.raises(ValueError, match="masking.enabled"):
        MaskedTrainer(cfg).load_checkpoint(checkpoint)


def test_sigreg_only_config():
    from pathlib import Path
    cfg = load_config(Path(__file__).parents[1] / "configs/sigreg_only.yaml")
    assert cfg.masking.enabled is False
    assert cfg.sigreg.weight == 1.0
