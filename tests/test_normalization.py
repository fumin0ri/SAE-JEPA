from __future__ import annotations

import math

import pytest
import torch

from sae_jepa.data import DataSource, TrainBatches, eval_batches, resolve_splits
from sae_jepa.frontends import PCAWhiteningFrontend
from sae_jepa.normalization import (
    check_normalization_matches,
    compute_train_statistics,
    denormalize,
    fit_pca_whitening,
    normalize,
)
from sae_jepa.synthetic import make_synthetic_manifest


def _train_rows(source: DataSource) -> torch.Tensor:
    return torch.cat([source.positions(p) for p in source.paths("train")]).double()


def test_statistics_match_direct_computation(manifest):
    source = DataSource(manifest)
    stats = compute_train_statistics(source)
    rows = _train_rows(source)
    mean = rows.mean(0)
    scale = math.sqrt((rows - mean).square().sum(1).mean() / rows.shape[1])
    assert torch.allclose(stats["mean"].double(), mean, atol=1e-4)
    assert stats["scale"] == pytest.approx(scale, rel=1e-6)
    assert stats["count"] == len(rows)
    x = normalize(rows.float(), stats["mean"], stats["scale"])
    assert float(x.square().sum(1).mean() / x.shape[1]) == pytest.approx(1.0, rel=1e-3)


def test_normalize_roundtrip(manifest):
    source = DataSource(manifest)
    stats = compute_train_statistics(source)
    h = source.positions(source.paths("validation")[0]).float()
    back = denormalize(normalize(h, stats["mean"], stats["scale"]), stats["mean"], stats["scale"])
    assert torch.allclose(back, h, atol=1e-3, rtol=1e-5)


def test_statistics_do_not_see_held_out_data(tmp_path):
    a = make_synthetic_manifest(tmp_path / "a", d_in=8, seed=5)
    b = make_synthetic_manifest(tmp_path / "b", d_in=8, seed=5, validation_offset=100.0)
    stats_a = compute_train_statistics(DataSource(a))
    stats_b = compute_train_statistics(DataSource(b))
    assert torch.equal(stats_a["mean"], stats_b["mean"])
    assert stats_a["scale"] == stats_b["scale"]


def test_leakage_and_split_checks(manifest):
    source = DataSource(manifest)
    stats = compute_train_statistics(source)
    assert set(stats["shards"]).isdisjoint(source.splits["validation"] + source.splits["test"])
    assert source.splits["test"], "holdout test split should be carved from validation"
    tampered = dict(stats, shards=stats["shards"] + [source.splits["validation"][0]])
    with pytest.raises(ValueError):
        check_normalization_matches(tampered, source)
    bad = {"train": {"shards": ["x.pt"]}, "validation": {"shards": ["x.pt"]}}
    with pytest.raises(ValueError):
        resolve_splits(bad, "none")


def test_train_batches_resume_exactly(manifest):
    source = DataSource(manifest)
    reference = TrainBatches(source, 16, seed=1, shards_per_window=2)
    expected = [next(reference) for _ in range(30)]
    first = TrainBatches(source, 16, seed=1, shards_per_window=2)
    for _ in range(11):
        next(first)
    resumed = TrainBatches(source, 16, seed=1, shards_per_window=2)
    resumed.load_state_dict(first.state_dict())
    for index in range(11, 30):
        assert torch.equal(next(resumed), expected[index])
    assert reference.epoch >= 1  # the test crosses an epoch boundary


def test_eval_batches_are_full_and_held_out(manifest):
    source = DataSource(manifest)
    batches = list(eval_batches(source, "validation", 16))
    assert batches and all(len(b) == 16 for b in batches)
    with pytest.raises(ValueError):
        next(eval_batches(source, "train", 16))


def test_pca_whitening_whitens_train_data(manifest):
    source = DataSource(manifest)
    stats = compute_train_statistics(source)
    frontend = PCAWhiteningFrontend(fit_pca_whitening(source, stats, epsilon=1e-8))
    y = frontend.encode_dense(_train_rows(source).float()).double()
    covariance = torch.cov(y.T, correction=0)
    assert torch.allclose(covariance, torch.eye(y.shape[1], dtype=y.dtype), atol=2e-3)
