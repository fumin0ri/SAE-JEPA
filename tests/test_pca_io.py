"""PCA I/O must preserve the old sample and half moments, not shuffled reads."""
import pytest
import torch

from sae_jepa.data import DataSource, eval_batches
from sae_jepa.normalization import pca_train_chunks
from sae_jepa.synthetic import make_lejepa_manifest


@pytest.mark.parametrize("maximum", [96, 100, 100000, 0])
@pytest.mark.parametrize("layout", ["torch", "safetensors"])
def test_ordered_reads_preserve_samples_and_half_moments(manifest, monkeypatch, maximum, layout, tmp_path):
    if layout == "safetensors":
        manifest = make_lejepa_manifest(tmp_path / "lejepa", d_in=16)
    source = DataSource(manifest, skip_leading_positions=1)
    batch_size, seed = 16, 3
    expected = [[], []]
    if maximum:
        for i, rows in enumerate(eval_batches(source, "train", batch_size,
                maximum // batch_size, seed=seed, allow_train=True)):
            expected[i % 2].append(rows)
    else:
        for i, entry in enumerate(source.paths("train")):
            expected[i % 2].append(source.positions(entry))
    expected = [torch.cat(rows).double() for rows in expected]
    actual = [[], []]
    calls = []
    gather = source.gather
    def recorded(entry, local):
        calls.append((source.paths("train").index(entry), local.clone()))
        assert len(local) <= batch_size
        assert torch.all(local[1:] > local[:-1])
        return gather(entry, local)
    monkeypatch.setattr(source, "gather", recorded)
    for rows, labels in pca_train_chunks(source, maximum, batch_size, seed):
        for half in (0, 1):
            actual[half].append(rows[labels == half])
    for reference, parts in zip(expected, actual):
        result = torch.cat(parts).double()
        # Exact row multiset, including membership of each diagnostic half.
        assert sorted(map(tuple, reference.tolist())) == sorted(map(tuple, result.tolist()))
        torch.testing.assert_close(result.T @ result, reference.T @ reference)
    # Every shard is visited once, with monotonically increasing row positions.
    for (owner_a, rows_a), (owner_b, rows_b) in zip(calls, calls[1:]):
        assert owner_b >= owner_a
        if owner_a == owner_b:
            assert rows_b[0] > rows_a[-1]


@pytest.mark.parametrize("maximum,batch", [(-1, 16), (100, 0), (16, 16)])
def test_invalid_pca_sampling_fails_early(manifest, maximum, batch):
    with pytest.raises(ValueError):
        list(pca_train_chunks(DataSource(manifest), maximum, batch, 3))
