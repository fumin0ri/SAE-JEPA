import json
from types import SimpleNamespace, ModuleType
import sys

import pytest
import torch

from conftest import tiny_config
from sae_jepa.data import eval_batches
from sae_jepa.masked import MaskedTrainer
from sae_jepa.norm_diagnostics import NormDiagnostics
from sae_jepa.synthetic import make_lejepa_manifest
from sae_jepa.verify_activations import capture, compare, parse_args, replay_inputs, run, StoredSequences


@pytest.fixture
def verification_data(tmp_path):
    st = pytest.importorskip('safetensors.torch')
    manifest = make_lejepa_manifest(tmp_path / 'data', d_in=8)
    raw = json.loads(manifest.read_text())
    # Known exact block output, with repeated IDs for matched-token controls.
    for shard in raw['shards']:
        file = manifest.parent / shard['file']
        tensors = st.load_file(str(file))
        ids = torch.arange(len(tensors['token_ids']), dtype=torch.int64) % 3
        st.save_file({'token_ids': ids, 'activations': (ids[:, None] + 3).expand(-1, 8).to(torch.bfloat16).contiguous()}, str(file))
    cfg = tiny_config(manifest, tmp_path / 'run')
    cfg.model.type = 'masked_sigreg_encoder'
    cfg.data.skip_leading_positions = 1
    trainer = MaskedTrainer(cfg)
    checkpoint = trainer.save_checkpoint()
    diag = NormDiagnostics()
    for h, meta in eval_batches(trainer.source, 'validation', 16, 3, seed=31337, with_metadata=True):
        y = torch.ones(len(h), 8)
        if not diag.rows:
            y[:2] *= 100
        diag.add(h, h, y, meta, trainer.source.paths('validation'))
    output = tmp_path / 'norms.json'
    files = diag.write(output, {'checkpoint': str(checkpoint), 'step': trainer.step,
        'split': 'validation', 'data_fingerprint': trainer.source.fingerprint})
    args = ['--summary', str(tmp_path / files['summary_path']), '--output', str(tmp_path / 'verification')]
    return trainer, args


def test_inspection_id_controls_and_loader(verification_data):
    trainer, common = verification_data
    report = run(parse_args(common))
    assert report['outlier_count'] == 2
    assert report['selected_outliers'] == 2
    assert report['loader_matches_all_selected']
    assert all(r['logged_h_norm_matches'] for r in report['samples'])
    assert any('same_token_control' in r['groups'] for r in report['samples'])
    assert all('context_token_ids' in r for r in report['samples'])
    with pytest.raises(ValueError, match='empty'):
        run(parse_args(common))


class Block(torch.nn.Module):
    def forward(self, x):
        return (x + 3,)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block()])
        self.config = SimpleNamespace(_commit_hash='a' * 40)
        self.seen = []

    def forward(self, input_ids, attention_mask, use_cache, position_ids=None):
        self.seen.append((input_ids.cpu().clone(), attention_mask.cpu().clone()))
        return self.layers[0](input_ids[..., None].expand(-1, -1, 8).float())


def install_fake_transformers(monkeypatch):
    model = FakeModel()
    module = ModuleType('transformers')
    module.AutoModel = SimpleNamespace(from_pretrained=lambda *a, **k: model)
    monkeypatch.setitem(sys.modules, 'transformers', module)
    monkeypatch.setattr('sae_jepa.verify_activations.importlib.metadata.version', lambda name: 'test')
    return model


def test_recompute_full_sequence_and_roundtrip(verification_data, monkeypatch):
    trainer, common = verification_data
    model = install_fake_transformers(monkeypatch)
    report = run(parse_args(common + ['--recompute', '--device', 'cpu', '--attention-mask', 'all-ones']))
    assert report['status'] == 'recomputation_complete'
    for r in report['samples']:
        assert r['recomputed_vs_stored']['exact_equal']
        assert r['storage_cast_vs_stored']['exact_equal']
        assert r['new_roundtrip']['exact_equal']
    assert len(model.seen) == len({(r['shard'], r['sequence']) for r in report['samples']})
    # Every forward uses a whole stored sequence, never context-radius truncation.
    lengths = {r['sequence_length'] for r in report['samples']}
    assert all(ids.shape[1] in lengths for ids, _ in model.seen)
    assert len(model.layers[0]._forward_hooks) == 0


def test_recompute_missing_mask_keeps_inspection(verification_data, monkeypatch):
    _, common = verification_data
    install_fake_transformers(monkeypatch)
    args = parse_args(common + ['--recompute', '--device', 'cpu'])
    with pytest.raises(ValueError, match='attention_mask missing'):
        run(args)
    from pathlib import Path
    report = json.loads((Path(args.output) / 'verification.json').read_text())
    assert report['status'] == 'recomputation_incomplete'
    assert report['samples'] and 'attention_mask missing' in report['error']


def test_detects_loader_misalignment(verification_data, monkeypatch):
    _, common = verification_data
    original = StoredSequences.loader_row
    monkeypatch.setattr(StoredSequences, 'loader_row', lambda self, row: original(self, row) + 1)
    report = run(parse_args(common))
    assert not report['loader_matches_all_selected']


def test_norm_provenance_mismatch_rejected(verification_data):
    _, common = verification_data
    from pathlib import Path
    path = Path(common[1])
    summary = json.loads(path.read_text())
    summary['provenance']['data_fingerprint'] = 'wrong'
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='provenance'):
        run(parse_args(common))


def test_missing_tokens_are_explicit(verification_data):
    trainer, common = verification_data
    from safetensors.torch import load_file, save_file
    for entry in trainer.source.paths('validation'):
        path = trainer.source.root / entry
        t = load_file(str(path))
        save_file({'activations': t['activations']}, str(path))
    report = run(parse_args(common))
    assert report['missing_token_id_count'] == report['sample_count']
    assert all(r['context_status'] == 'token_ids missing from shard' for r in report['samples'])


def test_compare_detects_vector_difference_even_with_equal_norms():
    result = compare(torch.tensor([1., 0.]), torch.tensor([0., 1.]))
    assert result['actual_sqnorm_per_dim'] == result['reference_sqnorm_per_dim']
    assert result['cosine'] == 0 and result['relative_l2_error'] > 1
    result = compare(torch.zeros(2), torch.zeros(2))
    assert result['exact_equal'] and result['relative_l2_error'] is None


def test_replay_preserves_stored_mask_and_positions():
    data = {'token_ids': torch.tensor([1, 2, 0]), 'attention_mask': torch.tensor([1, 1, 0]),
            'position_ids': torch.tensor([5, 6, 7])}
    result = replay_inputs(data, 'stored', 'cpu')
    assert result['attention_mask'].tolist() == [[1, 1, 0]]
    assert result['position_ids'].tolist() == [[5, 6, 7]]
    with pytest.raises(ValueError, match='token_ids missing'):
        replay_inputs({}, 'all-ones', 'cpu')


def test_capture_cleans_hook_on_failure():
    model = FakeModel()
    with pytest.raises(TypeError):
        capture(model, {'wrong': torch.ones(2)}, 0)
    assert not model.layers[0]._forward_hooks
