import json
from dataclasses import replace

import pytest
import torch

from conftest import tiny_config
from sae_jepa.data import TrainBatches
from sae_jepa.stage2 import (Stage2Config, Stage2Trainer, calibrate, evaluate, main,
                            preflight, tensor_hash, write_report)
from sae_jepa.topk import TopKSAE
from sae_jepa.train import Trainer


@pytest.fixture()
def front_checkpoint(manifest, tmp_path):
    t = Trainer(tiny_config(manifest, tmp_path / 'front', steps=2))
    t.train_step(False)
    return t.save_checkpoint()


def small_config():
    return Stage2Config(dictionary_size=32, k=4, steps=4, batch_size=16,
        warmup_steps=1, amp_dtype='none', shards_per_window=2,
        calibration_batches=2, eval_batch_size=16, eval_batches=2,
        validation_batches=2, validation_every=2, log_every=1, checkpoint_every=2)


def test_topk_nonnegative_budget_and_unit_decoder():
    sae = TopKSAE(8, 24, 3)
    x = torch.randn(16, 8)
    prediction, z = sae(x)
    assert (z >= 0).all() and ((z > 0).sum(1) <= 3).all()
    (prediction - x).square().mean().backward()
    sae.project_decoder_gradient()
    assert torch.max(abs((sae.decoder.weight * sae.decoder.weight.grad).sum(0))) < 1e-6
    torch.optim.Adam(sae.parameters()).step(); sae.normalize_decoder()
    torch.testing.assert_close(sae.decoder.weight.norm(dim=0), torch.ones(24))
    with torch.no_grad():
        sae.encoder.weight.zero_(); sae.encoder.bias.fill_(-1)
    assert torch.count_nonzero(sae(x)[1]) == 0


def test_train_calibration_freezing_and_resume(front_checkpoint, tmp_path):
    cfg = small_config()
    straight = Stage2Trainer(front_checkpoint, tmp_path / 'straight', cfg, 'cpu')
    original = tensor_hash(straight.frontend.state_dict())
    batches = TrainBatches(straight.source, cfg.batch_size, cfg.calibration_seed, cfg.shards_per_window)
    with torch.no_grad():
        y = torch.cat([straight.frontend.encode_dense(next(batches)) for _ in range(cfg.calibration_batches)])
    torch.testing.assert_close(straight.calibration['mean'], y.mean(0))
    assert straight.calibration['scale'] == pytest.approx(float((y.double() - y.double().mean(0)).square().mean().sqrt()))
    straight.run()
    split = Stage2Trainer(front_checkpoint, tmp_path / 'split', cfg, 'cpu')
    assert straight.initial_sha256 == split.initial_sha256
    split.run(max_steps=2)
    resumed = Stage2Trainer(front_checkpoint, tmp_path / 'split', cfg, 'cpu', resume=True)
    resumed.run()
    assert tensor_hash(straight.sae.state_dict()) == tensor_hash(resumed.sae.state_dict())
    assert torch.equal(straight.firing_counts, resumed.firing_counts)
    assert original == tensor_hash(straight.frontend.state_dict())
    assert all(p.grad is None and not p.requires_grad for p in straight.frontend.parameters())
    assert not straight.frontend.training
    with pytest.raises(ValueError, match='config changed'):
        Stage2Trainer(front_checkpoint, tmp_path / 'split', replace(cfg, lr=.1), 'cpu', resume=True)
    with pytest.raises(ValueError, match='config changed'):
        Stage2Trainer(front_checkpoint, tmp_path / 'split', replace(cfg, steps=5), 'cpu', resume=True)
    rows = [json.loads(x) for x in (tmp_path / 'split/metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows] == [1, 2, 3, 4]


def test_end_to_end_identity_and_zero_latent(front_checkpoint, tmp_path):
    t = Stage2Trainer(front_checkpoint, tmp_path / 'eval', small_config(), 'cpu')
    class Identity(torch.nn.Module):
        def forward(self, u):
            return u, torch.zeros(len(u), t.cfg.dictionary_size)
    result = evaluate(t.frontend, Identity(), t.calibration, t.source, t.cfg, t.device)
    assert result['latent']['fvu'] < 1e-12
    assert result['end_to_end']['fvu'] == pytest.approx(result['frontend']['fvu'], rel=1e-6)
    assert result['l0_mean'] == 0 and result['inactive_fraction'] == 1
    assert sum(result['feature_firing_counts']) == 0


def test_sweep_offline_eval_and_report(front_checkpoint, tmp_path):
    state = torch.load(front_checkpoint, weights_only=False)
    state['model']['encoder.0.weight'] += .01
    state['config']['sigreg']['weight'] = .001
    second = tmp_path / 'second.pt'; torch.save(state, second)
    cfg = small_config()
    preflight([front_checkpoint, second], cfg)
    cfg_path = tmp_path / 'settings.json'
    from dataclasses import asdict
    cfg_path.write_text(json.dumps(asdict(cfg)))
    output = tmp_path / 'comparison'
    common = ['sweep', '--checkpoints', str(front_checkpoint), str(second), '--output', str(output),
              '--config', str(cfg_path), '--device', 'cpu']
    main(common)
    main(common + ['--resume'])
    a = json.loads((output / 'model-00/eval-validation.json').read_text())
    b = json.loads((output / 'model-01/eval-validation.json').read_text())
    assert a['sample_sha256'] == b['sample_sha256']
    assert a['initial_sae_sha256'] == b['initial_sae_sha256']
    assert (output / 'report/validation.png').exists()
    offline = tmp_path / 'offline.json'
    main(['evaluate', '--checkpoint', str(output / 'model-00/checkpoints/latest.pt'),
          '--output', str(offline), '--device', 'cpu'])
    assert json.loads(offline.read_text()) == a
    b['sample_sha256'] = 'different'
    (output / 'model-01/eval-validation.json').write_text(json.dumps(b))
    with pytest.raises(ValueError, match='refusing report'):
        write_report(output)


def test_preflight_rejects_policy_mismatch(front_checkpoint, tmp_path):
    cfg = small_config()
    state = torch.load(front_checkpoint, weights_only=False)
    state['data_manifest']['burn_in_excluded'] += 1
    bad = tmp_path / 'bad.pt'; torch.save(state, bad)
    with pytest.raises(ValueError, match='exclusion'):
        preflight([front_checkpoint, bad], cfg)
    with pytest.raises(ValueError, match='duplicate'):
        preflight([front_checkpoint, front_checkpoint], cfg)
