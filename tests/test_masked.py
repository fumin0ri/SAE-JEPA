import json
from pathlib import Path

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


def sigreg_only_config(manifest, path):
    cfg = config(manifest, path)
    cfg.masking.enabled = False
    cfg.sigreg.weight = 1.0
    return cfg


def test_output_calibration_rescales_and_centers_only_the_last_layer(manifest, tmp_path):
    from sae_jepa.data import eval_batches, mix_seed
    from sae_jepa.masked import OUTPUT_INIT_BATCHES, OUTPUT_INIT_SEED_OFFSET

    base = MaskedTrainer(sigreg_only_config(manifest, tmp_path / "base"))
    cfg = sigreg_only_config(manifest, tmp_path / "calibrated")
    cfg.model.init_output_variance = 1.0
    calibrated = MaskedTrainer(cfg)
    info = calibrated.output_init
    assert base.output_init is None and "output_init" not in base.convention
    assert calibrated.convention["output_init"] == info
    assert info["samples"] == OUTPUT_INIT_BATCHES * cfg.eval.batch_size and info["gain"] > 0
    first, last = calibrated.model.encoder[0], calibrated.model.encoder[-1]
    assert torch.equal(first.weight, base.model.encoder[0].weight)
    assert torch.equal(first.bias, base.model.encoder[0].bias)
    torch.testing.assert_close(last.weight, info["gain"] * base.model.encoder[-1].weight)
    rows = torch.cat(list(eval_batches(
        calibrated.source, "train", cfg.eval.batch_size, OUTPUT_INIT_BATCHES,
        seed=mix_seed(cfg.train.seed, OUTPUT_INIT_SEED_OFFSET), allow_train=True)))
    with torch.no_grad():
        y = calibrated.model.encode_dense(rows).double()
    assert float(y.var(0, unbiased=False).mean()) == pytest.approx(1.0, rel=1e-4)
    assert float(y.mean(0).abs().max()) < 1e-4
    assert base.data.state_dict() == calibrated.data.state_dict()


def test_output_calibration_resumes_exactly(manifest, tmp_path):
    def make(path):
        cfg = sigreg_only_config(manifest, path)
        cfg.model.init_output_variance = 1.0
        return MaskedTrainer(cfg)

    straight = make(tmp_path / "straight")
    straight.run()
    make(tmp_path / "resume").run(max_steps=4)
    resumed = make(tmp_path / "resume")
    resumed.load_checkpoint(tmp_path / "resume/checkpoints/latest.pt")
    resumed.run()
    for key, value in straight.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key


def test_output_calibration_config_is_checked():
    with pytest.raises(ValueError, match="init_output_variance"):
        load_config(overrides=["model.init_output_variance=-1"])
    with pytest.raises(ValueError, match="init_output_variance"):
        load_config(overrides=["model.type=dense_sigreg_ae", "model.init_output_variance=1"])


def test_training_validation_logs_output_rank_from_step_zero(manifest, tmp_path):
    def validations(cfg):
        MaskedTrainer(cfg).run()
        rows = [json.loads(line) for line in
                (Path(cfg.train.output_dir) / "metrics.jsonl").read_text().splitlines()]
        return [row for row in rows if "validation" in row]

    logged = validations(sigreg_only_config(manifest, tmp_path / "on"))
    assert [row["step"] for row in logged] == [0, 4, 8]
    for row in logged:
        assert {"gaussian/cov_effective_rank", "gaussian/cov_participation_ratio",
                "gaussian/cov_eig_max"} <= row["validation"].keys()
        assert "reference/cov_effective_rank" not in row["validation"]
    masked = validations(config(manifest, tmp_path / "masked"))
    assert all("masked/cov_participation_ratio" in row["validation"] for row in masked)
    cfg = sigreg_only_config(manifest, tmp_path / "off")
    cfg.eval.training_covariance = False
    assert all("gaussian/cov_effective_rank" not in row["validation"] for row in validations(cfg))


def test_fit_readout_solves_ridge_and_feeds_stage2(manifest, tmp_path):
    from sae_jepa.data import eval_batches, mix_seed
    from sae_jepa.masked import READOUT_SEED_OFFSET
    from sae_jepa.stage2 import Stage2Trainer, preflight, write_report
    from test_stage2 import small_config

    cfg = config(manifest, tmp_path / "run")
    cfg.covariance.weight = 1.0
    cfg.covariance.sketch_dim = 4
    trainer = MaskedTrainer(cfg)
    trainer.run()
    checkpoint = tmp_path / "run/checkpoints/latest.pt"
    output = tmp_path / "readout/front.pt"
    main(["fit-readout", "--checkpoint", str(checkpoint), "--output", str(output),
          "--device", "cpu", "--batches", "4", "--ridge", "1e-3"])
    state = torch.load(output, weights_only=False)
    info = state["readout"]
    assert "optimizer" not in state and info["samples"] == 4 * cfg.eval.batch_size
    assert json.loads(output.with_suffix(".json").read_text()) == json.loads(json.dumps(info))
    rows = torch.cat(list(eval_batches(trainer.source, "train", cfg.eval.batch_size, 4,
                                       seed=mix_seed(cfg.train.seed, READOUT_SEED_OFFSET),
                                       allow_train=True)))
    model = trainer.model.eval()
    with torch.no_grad():
        y = model.encode_dense(rows).double()
    x = ((rows.float() - model.input_mean) / model.input_scale).double()
    yc, xc = y - y.mean(0), x - x.mean(0)
    cov_yy, cov_yx = yc.T @ yc / len(y), yc.T @ xc / len(y)
    penalty = 1e-3 * torch.trace(cov_yy) / y.shape[1]
    weight = torch.linalg.solve(cov_yy + penalty * torch.eye(y.shape[1], dtype=torch.float64), cov_yx)
    torch.testing.assert_close(state["model"]["readout.weight"].double(), weight.T, rtol=1e-4, atol=1e-5)
    fit = y @ weight + (x.mean(0) - y.mean(0) @ weight)
    assert info["train_fvu"] == pytest.approx(float((fit - x).square().sum() / xc.square().sum()), rel=1e-6)
    assert 0 < info["validation_fvu"]
    loaded, _ = load_checkpoint_model(output, torch.device("cpu"))
    with torch.no_grad():
        torch.testing.assert_close(loaded.decode_normalized(loaded.encode_dense(rows)).double(),
                                   fit, rtol=1e-4, atol=1e-4)
    with pytest.raises(ValueError, match="exists"):
        main(["fit-readout", "--checkpoint", str(checkpoint), "--output", str(output), "--device", "cpu"])
    with pytest.raises(ValueError, match="already has a readout"):
        main(["fit-readout", "--checkpoint", str(output), "--output", str(tmp_path / "again.pt"),
              "--device", "cpu"])
    s2 = small_config()
    assert preflight([output], s2)[0]["covariance_weight"] == 1.0
    root = tmp_path / "stage2"
    Stage2Trainer(output, root / "model-00", s2, "cpu").run()
    result = json.loads((root / "model-00/eval-validation.json").read_text())
    assert result["frontend_covariance_weight"] == 1.0 and result["frontend_readout"]["samples"] == 64
    assert result["frontend"]["fvu"] == pytest.approx(info["validation_fvu"], rel=.5)
    write_report(root)
    assert "| 1 |" in (root / "report/validation.md").read_text(encoding="utf-8")
    dense = Trainer(tiny_config(manifest, tmp_path / "dense", steps=2))
    dense.train_step(False)
    rows = preflight([dense.save_checkpoint(), output], s2)
    assert [row["covariance_weight"] for row in rows] == [0.0, 1.0]
