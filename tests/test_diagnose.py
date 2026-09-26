import json

import numpy as np
import pytest
import torch

from conftest import tiny_config
from sae_jepa.config import SIGRegConfig
from sae_jepa.data import LEJEPA_FORMAT
from sae_jepa.diagnose import (TokenContexts, diagnose_subset, keep_without_largest,
                               main, reference_mask)
from sae_jepa.train import Trainer


def test_trimmed_covariance_matches_direct_and_preserves_rng():
    g = torch.Generator().manual_seed(21)
    y = torch.randn(64, 4, generator=g)
    y[4] *= 100
    reference = torch.randn(64, 4, generator=g)
    keep = keep_without_largest(y.square().mean(1), 1 / 64)
    assert not keep[4] and keep.sum() == 63
    ref_keep = reference_mask(reference, keep, 16, True)
    assert [int(m.sum()) for m in ref_keep.split(16)] == [int(m.sum()) for m in keep.split(16)]
    before = torch.get_rng_state().clone()
    result, spectra = diagnose_subset(y, reference, keep, ref_keep, batch_size=16,
            device=torch.device('cpu'), sigreg_cfg=SIGRegConfig(), seeds=[901, 902], projections=8)
    assert torch.equal(before, torch.get_rng_state())
    centered = y[keep].double() - y[keep].double().mean(0)
    expected = torch.linalg.eigvalsh(centered.T @ centered / len(centered)).flip(0)
    torch.testing.assert_close(spectra['model'].double(), expected, atol=1e-6, rtol=1e-6)
    assert result['count'] == 63
    assert result['projection_seeds'][0]['model'] != result['projection_seeds'][1]['model']
    assert result['seed_summary']['reference']['w2_sq']['mean'] > 0


def test_context_ids_respect_sequence_boundary_and_global_sequence_index(tmp_path):
    # Minimal safetensors fixture with I32 IDs; no safetensors dependency needed.
    ids = np.arange(20, dtype=np.int32)
    header = json.dumps({'token_ids': {'dtype': 'I32', 'shape': [20], 'data_offsets': [0, 80]}}).encode()
    (tmp_path / 'shard.safetensors').write_bytes(len(header).to_bytes(8, 'little') + header + ids.tobytes())
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'shards': [{'file': 'shard.safetensors', 'sequences': [
        {'offset': 0, 'length': 8}, {'offset': 8, 'length': 12, 'document_id': 'doc', 'segment_index': 3}]}]}))
    from types import SimpleNamespace
    source = SimpleNamespace(format=LEJEPA_FORMAT, manifest_path=manifest, root=tmp_path)
    context = TokenContexts(source, 3).get('shard.safetensors#rows=1:2', 1, 1)
    assert context['context_token_ids'] == [8, 9, 10, 11, 12]
    assert context['token_id'] == 9 and context['target_index'] == 1
    assert context['document_id'] == 'doc'


def test_cli_shares_samples_is_repeatable_and_rejects_mismatch(manifest, tmp_path):
    trainer = Trainer(tiny_config(manifest, tmp_path / 'run'))
    checkpoint = trainer.save_checkpoint()
    common = ['--checkpoints', str(checkpoint), str(checkpoint), '--device', 'cpu',
              '--batch-size', '16', '--batches', '3', '--projections', '4',
              '--seeds', '901', '902', '--trim-fractions', '0.1', '--top', '2']
    before = torch.get_rng_state().clone()
    for name in ['first', 'second']:
        main(common + ['--output', str(tmp_path / name)])
    assert torch.equal(before, torch.get_rng_state())
    first = json.loads((tmp_path / 'first/model-00.json').read_text())
    other = json.loads((tmp_path / 'first/model-01.json').read_text())
    repeat = json.loads((tmp_path / 'second/model-00.json').read_text())
    assert first == repeat
    assert first['subsets'] == other['subsets']
    assert first['sample_sha256'] == other['sample_sha256']
    assert first['subsets']['full']['count'] == 48
    assert first['subsets']['trim_input_0.1']['count'] == 43
    assert first['subsets']['trim_input_0.1']['removed_indices'] == other['subsets']['trim_input_0.1']['removed_indices']
    assert (tmp_path / 'first/spectra.png').is_file()
    with pytest.raises(ValueError, match='empty'):
        main(common + ['--output', str(tmp_path / 'first')])
    state = torch.load(checkpoint, weights_only=False)
    state['data_manifest']['burn_in_excluded'] += 1
    bad = tmp_path / 'bad.pt'; torch.save(state, bad)
    with pytest.raises(ValueError, match='exclusion'):
        main(['--checkpoints', str(bad), '--output', str(tmp_path / 'bad'), '--device', 'cpu'])


def test_stable_ties_and_invalid_trim():
    assert keep_without_largest(torch.ones(10), .2).tolist() == [False, False] + [True] * 8
    with pytest.raises(ValueError):
        keep_without_largest(torch.ones(2), .5)


def test_trim_does_not_hide_broad_low_rank_structure():
    g = torch.Generator().manual_seed(17)
    y = torch.randn(128, 1, generator=g).repeat(1, 8)
    y[0] *= 100
    reference = torch.randn(128, 8, generator=g)
    keep = keep_without_largest(y.square().mean(1), .01)
    result, _ = diagnose_subset(y, reference, keep, keep, batch_size=32,
            device=torch.device('cpu'), sigreg_cfg=SIGRegConfig(), seeds=[901], projections=4)
    assert result['covariance']['cov_effective_rank'] == pytest.approx(1., abs=1e-5)
    assert result['reference_covariance']['cov_effective_rank'] > 7
