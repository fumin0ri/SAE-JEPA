import json
from dataclasses import asdict

import pytest
import torch

from conftest import tiny_config
from sae_jepa.baselines import BaselineFrontend, build_baseline
from sae_jepa.data import DataSource
from sae_jepa.stage2 import (Stage2Config, Stage2Trainer, build_front, evaluate,
                            load_front, main, preflight, source_for, tensor_hash)
from sae_jepa.train import Trainer


def settings():
    return Stage2Config(dictionary_size=32, k=4, steps=3, batch_size=16,
        warmup_steps=1, amp_dtype='none', shards_per_window=2,
        calibration_batches=2, eval_batch_size=16, eval_batches=2,
        validation_every=0, log_every=1, checkpoint_every=1)


@pytest.fixture()
def baselines(manifest, tmp_path):
    cfg = tiny_config(manifest, tmp_path / 'reference', steps=1)
    cfg.data.skip_leading_positions = 4
    ref = Trainer(cfg).save_checkpoint()
    output = tmp_path / 'baselines'
    main(['prepare-baselines', '--reference-checkpoint', str(ref), '--output', str(output),
          '--maximum-positions', '0', '--chunk-size', '31'])
    return ref, output / 'raw.pt', output / 'zca.pt'


def test_transforms_and_train_only_covariance(baselines):
    ref, raw_path, zca_path = baselines
    raw, raw_state = load_front(raw_path)
    zca, zca_state = load_front(zca_path)
    source = source_for(raw_state)
    assert source.burn_in >= 4
    h = torch.cat([source.positions(p) for p in source.paths('train')])
    x = (h - raw.input_mean) / raw.input_scale
    torch.testing.assert_close(raw.encode_dense(h), x)
    z = zca.encode_dense(h).double()
    torch.testing.assert_close(z.mean(0), torch.zeros(h.shape[1], dtype=torch.float64), atol=2e-5, rtol=0)
    covariance = torch.cov(x.double().T, correction=0)
    w = zca.whitening_matrix.double()
    expected = w @ covariance @ w.T
    torch.testing.assert_close(torch.cov(z.T, correction=0), expected, atol=2e-5, rtol=2e-5)
    # Regularized whitening: W (C + eps I) W^T = I, not exactly W C W^T = I.
    eps = zca_state['baseline_provenance']['whitening']['epsilon']
    torch.testing.assert_close(expected + eps * w @ w.T, torch.eye(len(w), dtype=torch.float64),
                               atol=2e-5, rtol=2e-5)
    for model, state in [(raw, raw_state), (zca, zca_state)]:
        torch.testing.assert_close(model.denormalize(model.decode_normalized(model.encode_dense(h))), h.float(),
                                   atol=2e-5, rtol=2e-5)
        assert not list(model.parameters())
        assert tensor_hash(build_front(state, 'cpu').state_dict()) == state['sha256']
        assert state['config']['data']['input_whitening_path'] is None
    assert zca_state['baseline_provenance']['whitening']['count'] == len(h)
    preflight([ref, raw_path, zca_path], settings())


def test_prepare_never_reads_heldout_and_is_reproducible(baselines, tmp_path, monkeypatch):
    ref, _, _ = baselines
    original = DataSource.gather
    def train_only(self, entry, indices):
        assert entry in self.paths('train')
        return original(self, entry, indices)
    monkeypatch.setattr(DataSource, 'gather', train_only)
    def no_full_read(*args, **kwargs):
        raise AssertionError('bounded fit must gather only sampled train positions')
    monkeypatch.setattr(DataSource, 'positions', no_full_read)
    for folder in ['one', 'two']:
        main(['prepare-baselines', '--reference-checkpoint', str(ref), '--output', str(tmp_path / folder),
              '--maximum-positions', '64', '--sample-seed', '15'])
    a = load_front(tmp_path / 'one/zca.pt')[1]
    b = load_front(tmp_path / 'two/zca.pt')[1]
    assert a['sha256'] == b['sha256']
    assert a['baseline_provenance']['whitening']['count'] == 64
    with pytest.raises(ValueError, match='empty'):
        main(['prepare-baselines', '--reference-checkpoint', str(ref), '--output', str(tmp_path / 'one')])


@pytest.mark.parametrize('index', [1, 2])
def test_baseline_resume_and_self_contained_evaluation(baselines, tmp_path, index):
    path = baselines[index]
    cfg = settings()
    straight = Stage2Trainer(path, tmp_path / 'straight', cfg, 'cpu')
    straight.run()
    interrupted = Stage2Trainer(path, tmp_path / 'split', cfg, 'cpu')
    interrupted.run(max_steps=1)
    resumed = Stage2Trainer(path, tmp_path / 'split', cfg, 'cpu', resume=True)
    resumed.run()
    assert tensor_hash(straight.sae.state_dict()) == tensor_hash(resumed.sae.state_dict())
    assert straight.initial_sha256 == resumed.initial_sha256
    class Identity(torch.nn.Module):
        def forward(self, u):
            return u, torch.zeros(len(u), cfg.dictionary_size)
    metrics = evaluate(straight.frontend, Identity(), straight.calibration,
                       straight.source, cfg, torch.device('cpu'))
    assert metrics['end_to_end']['fvu'] < 1e-10
    path.unlink()  # Standalone evaluation must not reopen its original front-end.
    offline = tmp_path / 'offline.json'
    main(['evaluate', '--checkpoint', str(tmp_path / 'straight/checkpoints/latest.pt'),
          '--output', str(offline), '--device', 'cpu'])
    assert json.loads(offline.read_text()) == json.loads((tmp_path / 'straight/eval-validation.json').read_text())


def test_mixed_sweep_same_samples_initialization_and_report(baselines, tmp_path):
    config_file = tmp_path / 'settings.json'
    config_file.write_text(json.dumps(asdict(settings())))
    root = tmp_path / 'sweep'
    args = ['sweep', '--checkpoints', *map(str, baselines), '--config', str(config_file),
            '--output', str(root), '--device', 'cpu']
    main(args)
    main(args + ['--resume'])
    rows = [json.loads((root / f'model-{i:02d}/eval-validation.json').read_text()) for i in range(3)]
    assert len({r['sample_sha256'] for r in rows}) == 1
    assert len({r['initial_sae_sha256'] for r in rows}) == 1
    assert [r['frontend_type'] for r in rows] == ['dense_sigreg_ae', 'raw', 'zca']
    report = (root / 'report/validation.md').read_text()
    assert 'Raw' in report and 'ZCA whitening' in report


def test_reject_corrupt_or_changed_baseline(baselines, tmp_path):
    _, path, _ = baselines
    state = torch.load(path, weights_only=False)
    state['normalization']['mean'] = state['normalization']['mean'] + 1
    with pytest.raises(ValueError, match='normalization differ'):
        build_baseline(state)
    with pytest.raises(ValueError, match='missing whitening'):
        BaselineFrontend('zca', state['normalization'])
    state = torch.load(path, weights_only=False)
    trainer = Stage2Trainer(path, tmp_path / 'resume', settings(), 'cpu')
    trainer.run(max_steps=1)
    state['model']['input_mean'] += 1
    state['normalization']['mean'] = state['model']['input_mean'].clone()
    changed = tmp_path / 'changed.pt'
    torch.save(state, changed)
    with pytest.raises(ValueError, match='checksum'):
        load_front(changed)
    state['sha256'] = tensor_hash(state['model'])
    repaired_hash = tmp_path / 'changed-valid-hash.pt'
    torch.save(state, repaired_hash)
    with pytest.raises(ValueError, match='changed'):
        Stage2Trainer(repaired_hash, tmp_path / 'resume', settings(), 'cpu', resume=True)


def test_pca_paper_covariance_inverse_and_activation_loss(baselines, tmp_path):
    ref, _, _ = baselines
    output = tmp_path / 'pca'
    main(['prepare-baselines', '--reference-checkpoint', str(ref), '--output', str(output),
          '--kinds', 'pca', '--maximum-positions', '0'])
    path = output / 'pca.pt'
    model, state = load_front(path)
    source = source_for(state)
    h = torch.cat([source.positions(p) for p in source.paths('train')])
    x = (h.double() - model.input_mean.double()) / model.input_scale.double()
    covariance = torch.cov(x.T)
    w = model.whitening_matrix.double()
    eps = state['baseline_provenance']['whitening']['epsilon']
    torch.testing.assert_close(w.T @ (covariance + eps * torch.eye(len(w))) @ w,
                               torch.eye(len(w), dtype=torch.float64), atol=2e-5, rtol=2e-5)
    z = model.encode_dense(h)
    torch.testing.assert_close(model.denormalize(model.decode_normalized(z)), h.float(), atol=2e-5, rtol=2e-5)
    cfg = settings()
    cfg.reconstruction_space = 'activation'
    trainer = Stage2Trainer(path, tmp_path / 'pca-train', cfg, 'cpu')
    batch = next(trainer.data)
    trainer.data = iter([batch])
    with torch.no_grad():
        u = (model.encode_dense(batch) - trainer.mean) / trainer.calibration['scale']
        prediction, _ = trainer.sae(u)
        h_hat = model.denormalize(model.decode_normalized(
            prediction * trainer.calibration['scale'] + trainer.mean))
        expected_loss = (h_hat - batch).square().mean().item()
    metrics = trainer.train_step()
    assert metrics['loss'] == pytest.approx(expected_loss)
    # Warmup starts at zero LR; gradients still must pass through the inverse.
    assert metrics['gradient_norm'] > 0
    assert all(torch.isfinite(p.grad).all() for p in trainer.sae.parameters() if p.grad is not None)
