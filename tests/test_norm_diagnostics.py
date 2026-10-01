import json

import pytest
import torch

from conftest import tiny_config
from sae_jepa.data import eval_batches
from sae_jepa.masked import MaskedTrainer, evaluate_masked, main
from sae_jepa.norm_diagnostics import NormDiagnostics, _correlation, _ranks


def test_known_amplification_and_outlier_overlap(tmp_path):
    diag = NormDiagnostics(.25)
    x = torch.tensor([[1.], [2.], [3.], [4.]])
    y = torch.tensor([[100., 100.], [2., 2.], [3., 3.], [4., 4.]])
    meta = {"entry": torch.zeros(4, dtype=torch.long), "sequence": torch.arange(4),
            "position": torch.arange(4) + 7}
    diag.add(x * 2, x, y, meta, ["shard.pt"])
    summary = diag.summary()
    view = summary["views"]["y"]
    assert summary["dimensions"] == {"h": 1, "x": 1, "y": 2}
    assert view["input_comparisons"]["x"]["input_top_overlap_count"] == 0
    assert view["top_samples"][0]["sample_index"] == 0
    assert view["top_samples"][0]["y_over_x_energy_ratio"] == 10000
    assert view["top_samples"][0]["position"] == 7
    assert view["output_top_energy_fraction"] == pytest.approx(10000 / 10029)
    assert view["groups"]["output_top"]["x_sqnorm_per_dim"]["mean"] == 1
    paths = diag.write(tmp_path / "eval.json", {"split": "validation"})
    rows = [json.loads(s) for s in (tmp_path / paths["samples_path"]).read_text().splitlines()]
    assert rows == diag.rows
    assert json.loads((tmp_path / paths["summary_path"]).read_text())["count"] == 4


def test_ties_zero_inputs_and_nonfinite():
    torch.testing.assert_close(_ranks(torch.tensor([4., 2., 2., 1.])),
                               torch.tensor([4., 2.5, 2.5, 1.], dtype=torch.float64))
    assert _correlation(torch.ones(4), torch.arange(4.)) is None
    diag = NormDiagnostics(.5)
    meta = {k: torch.zeros(4, dtype=torch.long) for k in ["entry", "sequence", "position"]}
    zero = torch.zeros(4, 2)
    diag.add(zero, zero, zero, meta, ["a"])
    s = diag.summary()["views"]["y"]
    assert s["undefined_energy_ratio_count"] == 4
    assert s["energy_ratio"] is None
    assert s["output_top_energy_fraction"] is None
    assert [r["sample_index"] for r in s["top_samples"]] == [0, 1]
    json.dumps(diag.summary(), allow_nan=False)
    with pytest.raises(ValueError, match="nonfinite"):
        diag.add(zero + float('inf'), zero, zero, meta, ["a"])


@pytest.mark.parametrize("enabled", [True, False])
def test_paired_evaluation_preserves_metrics_rng_and_sample_identity(manifest, tmp_path, enabled):
    cfg = tiny_config(manifest, tmp_path / "run")
    cfg.model.type = "masked_sigreg_encoder"
    cfg.masking.enabled = enabled
    trainer = MaskedTrainer(cfg)
    before = torch.get_rng_state()
    baseline = trainer.validate(True)
    diag = NormDiagnostics()
    actual = evaluate_masked(trainer.model, trainer.source, cfg, torch.device('cpu'), norm_diagnostics=diag)
    sa, sb = actual.pop('_spectra'), baseline.pop('_spectra')
    assert actual == baseline
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert torch.equal(before, torch.get_rng_state())
    assert trainer.model.training
    assert len(diag.rows) == actual['positions']
    offset = 0
    for h, meta in eval_batches(trainer.source, 'validation', cfg.eval.batch_size, cfg.eval.batches,
                                seed=cfg.eval.sample_seed, with_metadata=True):
        for i in range(len(h)):
            row = diag.rows[offset + i]
            assert row['h_sqnorm_per_dim'] == pytest.approx(float(h[i].double().square().mean()))
            assert row['shard'] == trainer.source.paths('validation')[int(meta['entry'][i])]
            assert row['sequence'] == int(meta['sequence'][i])
            assert row['position'] == int(meta['position'][i])
        offset += len(h)
    assert ('masked_y_sqnorm_per_dim' in diag.rows[0]) == enabled
    checkpoint = trainer.save_checkpoint()
    output = tmp_path / 'norm-eval.json'
    main(['evaluate', '--checkpoint', str(checkpoint), '--device', 'cpu', '--norm-diagnostics',
          '--output', str(output)])
    result = json.loads(output.read_text())
    summary = json.loads((tmp_path / result['norm_diagnostics']['summary_path']).read_text())
    assert summary['count'] == actual['positions']
    assert summary['provenance']['masking_enabled'] == enabled
    assert result['gaussian/heldout_sigreg'] == actual['gaussian/heldout_sigreg']


@pytest.mark.parametrize('fraction', [0, 1, -.1, float('nan')])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        NormDiagnostics(fraction)
