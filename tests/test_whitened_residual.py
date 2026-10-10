import pytest
import torch

from conftest import tiny_config
from sae_jepa.config import load_config
from sae_jepa.evaluate import load_checkpoint_model
from sae_jepa.input_whitening import fit
from sae_jepa.reporting import collect, write_report
from sae_jepa.stage2 import Stage2Config, Stage2Trainer, load_front, main, preflight
from sae_jepa.train import Trainer


def settings():
    return Stage2Config(dictionary_size=32, k=4, steps=3, batch_size=16,
        warmup_steps=1, amp_dtype='none', shards_per_window=2,
        calibration_batches=2, eval_batch_size=16, eval_batches=2,
        validation_every=0, log_every=1, checkpoint_every=1, reconstruction_space='activation')


@pytest.fixture()
def whitening(manifest, tmp_path):
    reference = Trainer(tiny_config(manifest, tmp_path / 'reference', steps=1))
    ref = reference.save_checkpoint()
    output = tmp_path / 'baselines'
    main(['prepare-baselines', '--reference-checkpoint', str(ref), '--output', str(output),
          '--maximum-positions', '0', '--chunk-size', '31', '--kinds', 'pca', 'zca'])
    return tmp_path / 'reference' / 'normalization.pt', output / 'pca.pt', output / 'zca.pt'


def config(manifest, output, normalization, whitening, weight=0.1, steps=8):
    cfg = tiny_config(manifest, output, weight=weight, steps=steps)
    cfg.model.type = 'whitened_residual_ae'
    cfg.data.normalization_path = str(normalization)
    cfg.data.input_whitening_path = str(whitening)
    cfg.validate()
    return cfg


def train_rows(trainer):
    return torch.cat([trainer.source.positions(p) for p in trainer.source.paths('train')])


def test_starts_exactly_at_the_whitening_baseline(manifest, tmp_path, whitening):
    normalization, pca_path, _ = whitening
    trainer = Trainer(config(manifest, tmp_path / 'run', normalization, pca_path))
    dense = Trainer(tiny_config(manifest, tmp_path / 'dense'))
    pca, _ = load_front(pca_path)
    h = train_rows(trainer)
    out = trainer.model(h)
    torch.testing.assert_close(out['y'], pca.encode_dense(h), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out['x_hat'], out['x'], atol=1e-4, rtol=1e-4)
    assert not trainer.model.encoder[-1].weight.any()
    # Same seed: same first-layer initialization and data order as the dense AE.
    assert torch.equal(trainer.model.encoder[0].weight, dense.model.encoder[0].weight)
    assert torch.equal(next(trainer.data), next(dense.data))
    assert 'whitening_matrix' not in dict(trainer.model.named_parameters())
    assert trainer.convention['input_transform'] == 'pca'
    # A whitening fit file gives the same transform as the baseline front-end.
    stats = fit(trainer.source, trainer.normalization, epsilon=1e-4, chunk_size=31, kind='pca')
    torch.save(stats, tmp_path / 'fit.pt')
    from_fit = Trainer(config(manifest, tmp_path / 'fit-run', normalization, tmp_path / 'fit.pt'))
    torch.testing.assert_close(from_fit.model(h)['y'], out['y'], atol=1e-5, rtol=1e-5)


def test_paired_diagnostic_restores_initial_without_changing_checkpoint(manifest, tmp_path, whitening):
    from sae_jepa.paired_whitening import diagnose
    from sae_jepa.stage2 import tensor_hash
    normalization, pca_path, _ = whitening
    trainer = Trainer(config(manifest, tmp_path / 'paired', normalization, pca_path))
    checkpoint = trainer.save_checkpoint()
    initial = diagnose(checkpoint, tmp_path / 'initial-diagnostic', device='cpu', batches=2)
    assert all(v['delta'] == 0 for v in initial['metrics'].values())
    with torch.no_grad():
        trainer.model.encoder[-1].bias.add_(0.2)
    checkpoint = trainer.save_checkpoint()
    original_hash = tensor_hash(trainer.model.state_dict())
    result = diagnose(checkpoint, tmp_path / 'changed-diagnostic', device='cpu', batches=2)
    assert result['sample_sha256'] == initial['sample_sha256']
    for k, v in result['metrics'].items():
        assert v['before'] == initial['metrics'][k]['before']
    assert result['metrics']['gaussian/heldout_sigreg']['delta'] != 0
    saved, _ = load_checkpoint_model(checkpoint, torch.device('cpu'))
    assert tensor_hash(saved.state_dict()) == original_hash


def test_trains_resumes_and_is_a_stage2_frontend(manifest, tmp_path, whitening):
    normalization, pca_path, zca_path = whitening
    make = lambda name: Trainer(config(manifest, tmp_path / name, normalization, pca_path))
    full = make('full')
    final = full.run()
    assert final['step'] == 8 and final['reconstruction/fvu'] < 1
    assert full.model.encoder[-1].weight.any()  # SIGReg moved y away from z
    write_report(collect(tmp_path / 'full'), tmp_path / 'report', run_root=tmp_path / 'full')
    assert (tmp_path / 'report' / 'validation.md').exists()
    part = make('part')
    part.run(max_steps=4)
    checkpoint = tmp_path / 'part/checkpoints/latest.pt'
    resumed = make('part')
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    for key, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key])

    other = Trainer(config(manifest, tmp_path / 'part', normalization, zca_path))
    with pytest.raises(ValueError, match='whitening'):
        other.load_checkpoint(checkpoint)

    latest = tmp_path / 'full/checkpoints/latest.pt'
    model, state = load_checkpoint_model(latest, torch.device('cpu'))
    h = train_rows(full)
    assert torch.equal(model.encode_dense(h), full.model.encode_dense(h))
    assert state['input_whitening']['kind'] == 'pca'
    preflight([latest, pca_path], settings())
    stage2 = Stage2Trainer(latest, tmp_path / 'stage2', settings(), 'cpu')
    stage2.run()
    assert stage2.step == 3


def test_configuration_errors(manifest, tmp_path, whitening):
    normalization, pca_path, _ = whitening
    with pytest.raises(ValueError, match='requires data.input_whitening_path'):
        load_config(None, [f'data.activation_manifest={manifest.as_posix()}',
                           'model.type=whitened_residual_ae'])
    cfg = config(manifest, tmp_path / 'run', normalization, pca_path)
    state = torch.load(normalization, weights_only=False)
    state['scale'] = float(state['scale']) * 2
    torch.save(state, tmp_path / 'other-normalization.pt')
    cfg.data.normalization_path = str(tmp_path / 'other-normalization.pt')
    with pytest.raises(ValueError, match='different scalar normalization'):
        Trainer(cfg)
    cfg = config(manifest, tmp_path / 'unknown', normalization, pca_path)
    cfg.data.input_whitening_path = str(tmp_path / 'unknown.pt')
    torch.save(torch.load(pca_path, weights_only=False) | {'format': 'unknown'}, cfg.data.input_whitening_path)
    with pytest.raises(ValueError, match='neither'):
        Trainer(cfg)
