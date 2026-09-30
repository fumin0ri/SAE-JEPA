from __future__ import annotations

import numpy as np
import pytest
import torch

from sae_jepa.data import (
    READ_THREADS_ENV,
    DataSource,
    TrainBatches,
    _safetensors_array,
    mix_seed,
    read_safetensors_rows,
)
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


def _original_batches(source, batch_size, seed, shards_per_window, count):
    """Batches produced by the original (copying) TrainBatches implementation."""
    paths = source.paths("train")
    windows = -(-len(paths) // shards_per_window)
    out, epoch = [], 0
    while len(out) < count:
        shard_order = torch.randperm(
            len(paths), generator=torch.Generator().manual_seed(mix_seed(seed, epoch))
        ).tolist()
        for window in range(windows):
            chosen = shard_order[window * shards_per_window : (window + 1) * shards_per_window]
            rows = torch.cat([source.positions(paths[i]) for i in chosen])
            order = torch.randperm(
                len(rows), generator=torch.Generator().manual_seed(mix_seed(seed, epoch, window))
            )
            for offset in range(0, len(rows) - batch_size + 1, batch_size):
                out.append(rows[order[offset : offset + batch_size]])
        epoch += 1
    return out[:count]


@pytest.mark.parametrize("layout", ["lejepa", "jepa"])
@pytest.mark.parametrize("k", [0, 1, 3])
@pytest.mark.parametrize("prefetch,threads", [(False, "1"), (True, "1"), (True, "4")])
def test_batches_match_original_implementation(tmp_path, monkeypatch, layout, k, prefetch, threads):
    monkeypatch.setenv(READ_THREADS_ENV, threads)
    make = make_lejepa_manifest if layout == "lejepa" else make_synthetic_manifest
    source = DataSource(make(tmp_path / "d", d_in=8), skip_leading_positions=k)
    batches = TrainBatches(source, 16, seed=5, shards_per_window=2, prefetch=prefetch)
    for expected in _original_batches(source, 16, 5, 2, 60):
        assert torch.equal(next(batches), expected)


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
