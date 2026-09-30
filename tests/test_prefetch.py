from __future__ import annotations

import numpy as np
import pytest
import torch

from sae_jepa.data import DataSource, TrainBatches, _safetensors_array, read_safetensors_rows
from sae_jepa.synthetic import make_lejepa_manifest, make_synthetic_manifest


def _memmap_rows(path, rows):
    array, _ = _safetensors_array(path)
    return torch.from_numpy(np.ascontiguousarray(array[rows.numpy()])).view(torch.bfloat16)


@pytest.mark.parametrize("k", [0, 1, 3])
def test_sequential_read_matches_memmap(tmp_path, k):
    path = make_lejepa_manifest(tmp_path / "d", d_in=8)
    source = DataSource(path, skip_leading_positions=k)
    for entry in source.paths("train"):
        file = source.root / entry
        rows = source._flat_rows(entry, None)
        assert torch.equal(read_safetensors_rows(file, rows), _memmap_rows(file, rows))
        sparse = rows[:: max(1, len(rows) // 3)]
        assert torch.equal(read_safetensors_rows(file, sparse), _memmap_rows(file, sparse))


@pytest.mark.parametrize("layout", ["lejepa", "jepa"])
def test_prefetch_yields_identical_batches_and_resumes(tmp_path, layout):
    make = make_lejepa_manifest if layout == "lejepa" else make_synthetic_manifest
    source = DataSource(make(tmp_path / "d", d_in=8), skip_leading_positions=1)
    plain = TrainBatches(source, 16, seed=3, shards_per_window=2, prefetch=False)
    fast = TrainBatches(source, 16, seed=3, shards_per_window=2)
    steps = 60  # spans several windows and epochs of the tiny dataset
    reference = [next(plain) for _ in range(steps)]
    for i in range(steps):
        assert torch.equal(next(fast), reference[i])
    assert fast.state_dict() == plain.state_dict()

    resumed = TrainBatches(source, 16, seed=3, shards_per_window=2)
    partial = TrainBatches(source, 16, seed=3, shards_per_window=2)
    for _ in range(steps // 2):
        next(partial)
    resumed.load_state_dict(partial.state_dict())
    for i in range(steps // 2, steps):
        assert torch.equal(next(resumed), reference[i])
