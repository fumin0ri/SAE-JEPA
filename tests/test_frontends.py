import json
from dataclasses import replace

import pytest
import torch

from conftest import tiny_config
from sae_jepa.data import DataSource
from sae_jepa.frontends import PCAWhiteningFrontend, load_frontend
from sae_jepa.normalization import (compute_train_statistics, fit_pca_whitening, pca_main,
                                    raw_main, resolve_pca_epsilon)
from sae_jepa.stage2 import (Stage2Config, Stage2Trainer, comparable_config, evaluate, load_front,
                             main, preflight)
from sae_jepa.train import Trainer


def small_config(**kw):
    cfg = Stage2Config(dictionary_size=32, k=4, steps=4, batch_size=16,
        warmup_steps=1, amp_dtype='none', shards_per_window=2,
        calibration_batches=2, eval_batch_size=16, eval_batches=2,
        validation_batches=2, validation_every=2, log_every=1, checkpoint_every=2)
    return replace(cfg, **kw)


@pytest.fixture()
def dense(manifest, tmp_path):
    t = Trainer(tiny_config(manifest, tmp_path / 'front', steps=2))
    t.train_step(False)
    return t.save_checkpoint()


@pytest.fixture()
def fronts(dense, tmp_path):
    raw, pca = tmp_path / 'raw.pt', tmp_path / 'pca.pt'
    raw_main(['--like', str(dense), '--output', str(raw)])
    pca_main(['--like', str(dense), '--output', str(pca), '--maximum-positions', '0',
              '--relative-epsilon', '1e-6'])
    return raw, pca


def test_pca_dewhitening_is_exact_inverse_and_fp32_under_autocast(manifest):
    source = DataSource(manifest)
    stats = compute_train_statistics(source)
    front = PCAWhiteningFrontend(fit_pca_whitening(source, stats, epsilon=1e-3))
    h = source.positions(source.paths('validation')[0]).float()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y = front.encode_dense(h)
        back = front.denormalize(front.decode_normalized(y))
    assert y.dtype == torch.float32
    torch.testing.assert_close(back, h, atol=1e-4, rtol=1e-4)


def test_pca_relative_epsilon_sampling_and_diagnostics(manifest):
    source = DataSource(manifest)
    stats = compute_train_statistics(source)
    full = fit_pca_whitening(source, stats, relative_epsilon=1e-2)
    assert full['epsilon'] == pytest.approx(1e-2 * float(full['eigenvalues'].clamp_min(0).mean()))
    assert resolve_pca_epsilon(full['eigenvalues'], 0.5, 1e-2) == 0.5
    d = full['diagnostics']
    assert d['count'] == source.split_positions('train') == sum(d['half_counts'])
    assert d['trace'] == pytest.approx(float(full['eigenvalues'].sum()), rel=1e-5)
    assert all(0 <= v <= 1 + 1e-9 for v in d['subspace_overlap'].values())
    assert d['subspace_overlap'][str(16)] == pytest.approx(1.0)  # all 16 dimensions
    sampled = fit_pca_whitening(source, stats, maximum_positions=100, batch_size=16, sample_seed=3)
    again = fit_pca_whitening(source, stats, maximum_positions=100, batch_size=16, sample_seed=3)
    assert sampled['count'] == 96 and torch.equal(sampled['eigenvalues'], again['eigenvalues'])
    assert sampled['sampling']['method'].startswith('uniform')
    with pytest.raises(ValueError, match='two batches'):
        fit_pca_whitening(source, stats, maximum_positions=20, batch_size=16)


def test_frontend_files_copy_data_policy_and_statistics(dense, fronts, tmp_path):
    raw, pca = fronts
    cfg = small_config()
    rows = preflight([dense, raw, pca], cfg)
    assert [r.get('kind') for r in rows] == [None, 'raw', 'pca']
    assert load_frontend('pca', pca).kind == 'pca' and load_frontend('raw', raw).kind == 'raw'
    with pytest.raises(ValueError, match='not'):
        load_frontend('raw', pca)
    with pytest.raises(SystemExit):
        raw_main(['--like', str(dense), '--output', str(raw)])
    state = torch.load(raw, weights_only=False)
    state['normalization']['mean'] = state['normalization']['mean'] + 1
    shifted = tmp_path / 'shifted.pt'; torch.save(state, shifted)
    with pytest.raises(ValueError, match='normalization'):
        preflight([dense, shifted], cfg)
    summary = json.loads(pca.with_suffix('.json').read_text())
    assert summary['data']['skip_leading_positions'] == 0 and 'condition_number' in summary['diagnostics']


def test_raw_loss_spaces_differ_only_by_calibration_scale(fronts, tmp_path):
    raw, _ = fronts
    latent = Stage2Trainer(raw, tmp_path / 'latent', small_config(), 'cpu')
    original = Stage2Trainer(raw, tmp_path / 'original', small_config(loss_space='original'), 'cpu')
    a, b = latent.train_step(), original.train_step()
    scale = latent.calibration['scale']
    assert b['normalized_input_mse'] == pytest.approx(a['normalized_latent_mse'] * scale**2, rel=1e-5)
    result = evaluate(latent.frontend, latent.sae, latent.calibration, latent.source, latent.cfg, latent.device)
    assert result['frontend']['fvu'] < 1e-12
    assert result['latent']['fvu'] == pytest.approx(result['end_to_end']['fvu'], rel=1e-5)


def test_pca_and_dense_original_loss_train_frozen_frontends(dense, fronts, tmp_path):
    _, pca = fronts
    for name, path in [('pca', pca), ('dense', dense)]:
        t = Stage2Trainer(path, tmp_path / name, small_config(loss_space='original'), 'cpu')
        before = {k: v.clone() for k, v in t.frontend.state_dict().items()}
        t.run()
        assert all(torch.equal(before[k], v) for k, v in t.frontend.state_dict().items())
        assert all(p.grad is None for p in t.frontend.parameters())
        result = json.loads((tmp_path / name / 'eval-validation.json').read_text())
        assert result['loss_space'] == 'original' and result['frontend_kind'] == name
        if name == 'pca':
            assert result['frontend']['fvu'] < 1e-8


def test_sweep_crosses_frontends_and_loss_spaces(dense, fronts, tmp_path):
    raw, pca = fronts
    cfg_path = tmp_path / 'settings.json'
    from dataclasses import asdict
    cfg_path.write_text(json.dumps(asdict(small_config())))
    output = tmp_path / 'comparison'
    common = ['sweep', '--checkpoints', str(raw), str(pca), str(dense), '--output', str(output),
              '--config', str(cfg_path), '--device', 'cpu', '--loss-spaces', 'latent', 'original']
    main(common)
    main(common + ['--resume'])
    results = [json.loads((output / f'model-{i:02d}/eval-validation.json').read_text()) for i in range(6)]
    assert [(r['frontend_kind'], r['loss_space']) for r in results] == [
        ('raw', 'latent'), ('raw', 'original'), ('pca', 'latent'), ('pca', 'original'),
        ('dense', 'latent'), ('dense', 'original')]
    assert len({r['initial_sae_sha256'] for r in results}) == 1
    report = (output / 'report/validation.md').read_text(encoding='utf-8')
    assert 'pca ε=' in report and '| original |' in report
    with pytest.raises(ValueError, match='changed'):
        main(common[:-1] + ['--resume'])
    offline = tmp_path / 'offline.json'
    main(['evaluate', '--checkpoint', str(output / 'model-03/checkpoints/latest.pt'),
          '--output', str(offline), '--device', 'cpu'])
    assert json.loads(offline.read_text()) == results[3]


def test_probe_compares_frontends_and_loss_spaces(dense, fronts, tmp_path):
    from sae_jepa.probe_data import checkpoint_info, file_hash, text_id
    from sae_jepa.probing import main as probe
    from test_probing import examples
    raw, pca = fronts
    checkpoints = []
    for name, path, space in [('raw', raw, 'latent'), ('pca', pca, 'original')]:
        checkpoint = Stage2Trainer(path, tmp_path / name, small_config(loss_space=space), 'cpu').save()
        state = torch.load(checkpoint, weights_only=False)
        state['data_manifest']['hook_point'] = f"block_output:{state['data_manifest']['layer']}"
        torch.save(state, checkpoint)
        checkpoints.append(str(checkpoint))
    rows = examples(); tasks = tmp_path / 'tasks.jsonl'
    tasks.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    cache = tmp_path / 'cache'; cache.mkdir()
    ids = [text_id(r['text']) for r in rows]
    torch.save({'ids': ids, 'h': torch.randn(len(ids), 3, 16, generator=torch.Generator().manual_seed(1)),
                'mask': torch.tensor([[False, True, True]] * len(ids))}, cache / 'batch.pt')
    (cache / 'manifest.json').write_text(json.dumps({'format': 'sae-jepa-probe-activations-v1',
        'signature': {'tasks_sha256': file_hash(tasks), 'identity': checkpoint_info(checkpoints[0])},
        'ids': ids, 'chunks': [{'file': 'batch.pt', 'sha256': file_hash(cache / 'batch.pt')}]}))
    probe(['evaluate', '--tasks', str(tasks), '--activations', str(cache), '--checkpoints', *checkpoints,
           '--ks', '1', '--cs', '1', '--max-iter', '20', '--device', 'cpu', '--output', str(tmp_path / 'p')])
    summary = (tmp_path / 'p/summary.md').read_text(encoding='utf-8')
    assert 'raw / latent' in summary and 'pca ε=' in summary and '/ original' in summary


def test_checkpoints_without_kind_or_loss_space_still_load(dense, tmp_path):
    t = Stage2Trainer(dense, tmp_path / 'old', small_config(), 'cpu')
    t.run()
    state = torch.load(t.save(), weights_only=False)
    del state['config']['loss_space'], state['frontend']['kind']
    old = tmp_path / 'old.pt'; torch.save(state, old)
    out = tmp_path / 'old.json'
    main(['evaluate', '--checkpoint', str(old), '--output', str(out), '--device', 'cpu'])
    result = json.loads(out.read_text())
    assert result['loss_space'] == 'latent' and result['frontend_label'].startswith('dense')
    assert comparable_config(state['config']) == comparable_config(
        {**state['config'], 'loss_space': 'original'})
    _, front = load_front(dense)
    assert front['kind'] == 'dense'
