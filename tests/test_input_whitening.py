import json

import pytest
import torch

from conftest import tiny_config
from sae_jepa.data import DataSource
from sae_jepa.input_whitening import fit, validate, main as fit_main
from sae_jepa.masked import MaskedTrainer, main
from sae_jepa.normalization import compute_train_statistics, save_normalization


def prepare(manifest, tmp_path):
    source = DataSource(manifest)
    normalization = compute_train_statistics(source)
    stats = fit(source, normalization, epsilon=1e-6, chunk_size=17)
    path = tmp_path / 'whitening.pt'
    torch.save(stats, path)
    return source, normalization, stats, path


def test_whitening_covariance_train_only_and_regularization(manifest, tmp_path):
    source, norm, w, _ = prepare(manifest, tmp_path)
    h = torch.cat([source.positions(p) for p in source.paths('train')]).double()
    x = (h-norm['mean'].double())/norm['scale']
    z = (x-w['center'].double()) @ w['matrix'].double()
    covariance = (z-z.mean(0)).T @ (z-z.mean(0))/len(z)
    eig = torch.linalg.eigvalsh(covariance).flip(0)
    torch.testing.assert_close(eig, w['expected_transformed_eigenvalues'], rtol=1e-4, atol=1e-5)
    assert z.mean(0).abs().max() < 1e-5
    # Held-out values may change arbitrarily without changing train-fit stats.
    for entry in source.paths('validation'):
        path = source.root / entry
        data = torch.load(path, weights_only=False)
        data['activations'] += 10000
        torch.save(data, path)
    again = fit(source, norm, epsilon=1e-6, chunk_size=17)
    assert torch.equal(w['matrix'], again['matrix'])
    bad = {**w, 'split': 'validation'}
    with pytest.raises(ValueError, match='train-fit'):
        validate(bad, source, norm)
    with pytest.raises(ValueError, match='excluded'):
        validate({**w, 'burn_in_excluded': source.burn_in+1}, source, norm)


def test_initialization_resume_and_self_contained_evaluation(manifest, tmp_path):
    source, norm, w, path = prepare(manifest, tmp_path)
    cfg = tiny_config(manifest, tmp_path/'white')
    cfg.model.type = 'masked_sigreg_encoder'
    cfg.masking.enabled = False
    cfg.eval.input_diagnostics = True
    baseline = MaskedTrainer(cfg)
    cfg.data.input_whitening_path = str(path)
    trainer = MaskedTrainer(cfg)
    for a, b in zip(baseline.model.parameters(), trainer.model.parameters()):
        assert torch.equal(a, b)
    assert torch.equal(next(baseline.data), next(trainer.data))
    assert 'whitening_matrix' not in dict(trainer.model.named_parameters())
    # Start fresh after the data-order assertion.
    trainer = MaskedTrainer(cfg)
    trainer.run(max_steps=4)
    checkpoint = trainer.output_dir/'checkpoints/latest.pt'
    resumed = MaskedTrainer(cfg)
    resumed.load_checkpoint(checkpoint)
    expected = trainer.run()
    actual = resumed.run()
    assert actual['input_transform'] == 'zca'
    assert 'encoder_input/cov_effective_rank' in actual
    assert actual['gaussian/heldout_sigreg'] == expected['gaussian/heldout_sigreg']
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key])
    path.unlink()
    output = tmp_path/'evaluation.json'
    main(['evaluate', '--checkpoint', str(checkpoint), '--device', 'cpu', '--output', str(output)])
    offline = json.loads(output.read_text())
    assert offline['input_transform'] == 'zca'
    assert offline['gaussian/heldout_sigreg'] == actual['gaussian/heldout_sigreg']
    changed = {**w, 'matrix': w['matrix']*2}
    torch.save(changed, path)
    with pytest.raises(ValueError, match='statistics changed'):
        MaskedTrainer(cfg).load_checkpoint(checkpoint)


def test_fit_cli_and_no_masked_mixing(manifest, tmp_path):
    source, norm, w, path = prepare(manifest, tmp_path)
    norm_path = tmp_path/'normalization.pt'
    save_normalization(norm, norm_path)
    output = tmp_path/'cli.pt'
    fit_main(['--activation-manifest', str(manifest), '--normalization', str(norm_path),
              '--output', str(output), '--maximum-positions', '32', '--chunk-size', '7'])
    report = json.loads(output.with_suffix('.json').read_text())
    assert report['count'] == 32 and report['sampling'] == 'uniform without replacement over usable train positions'
    cfg = tiny_config(manifest, tmp_path/'bad')
    cfg.model.type = 'masked_sigreg_encoder'
    cfg.data.input_whitening_path = str(path)
    with pytest.raises(ValueError, match='SIGReg-only'):
        cfg.validate()


@pytest.mark.parametrize('epsilon', [0, -1, float('nan')])
def test_bad_epsilon(manifest, epsilon):
    source = DataSource(manifest)
    with pytest.raises(ValueError):
        fit(source, compute_train_statistics(source), epsilon=epsilon)


def test_capped_fit_samples_all_train_positions_without_loading_full_shards(manifest, monkeypatch):
    import random
    source = DataSource(manifest)
    norm = compute_train_statistics(source)
    all_rows = torch.cat([source.positions(e) for e in source.paths('train')]).double()
    selected = sorted(random.Random(1729).sample(range(len(all_rows)), 100))
    expected = (all_rows[selected] - norm['mean'].double()) / norm['scale']
    def forbidden(*args, **kwargs):
        raise AssertionError('capped fit must gather selected positions only')
    monkeypatch.setattr(source, 'positions', forbidden)
    original = source.gather
    visited = []
    def gather(entry, rows):
        assert entry in source.paths('train')
        visited.append(entry)
        return original(entry, rows)
    monkeypatch.setattr(source, 'gather', gather)
    rng = torch.get_rng_state()
    w = fit(source, norm, maximum_positions=100, chunk_size=11)
    torch.testing.assert_close(w['center'].double(), expected.mean(0), rtol=1e-5, atol=1e-7)
    centered = expected-expected.mean(0)
    torch.testing.assert_close(w['train_eigenvalues'], torch.linalg.eigvalsh(centered.T @ centered/100).flip(0))
    assert len(set(visited)) > 1
    assert torch.equal(rng, torch.get_rng_state())
