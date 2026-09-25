from __future__ import annotations

import json

import pytest
import torch

from sae_jepa.config import EvalConfig
from sae_jepa.data import DataSource, eval_batches, read_safetensors_rows
from sae_jepa.evaluate import _Moments, _OutlierDiagnostics
from sae_jepa.synthetic import make_lejepa_manifest

from test_review_fixes import id_shard, write_manifest


def test_eval_metadata_locates_rows_in_jepa_layout(tmp_path):
    length, burn_in = 20, 3
    gen = torch.Generator().manual_seed(0)
    shards = [id_shard(s, 6, length, torch.randint(burn_in + 1, length + 1, (6,), generator=gen))
              for s in range(3)]
    source = DataSource(write_manifest(tmp_path, {"train": shards[:1], "validation": shards},
                                       length, burn_in), test_split="none")
    for rows, meta in eval_batches(source, "validation", 8, 0, seed=2, with_metadata=True):
        ids = rows[:, 0].long()
        shard = torch.tensor([int(source.paths("validation")[e].split("-")[-1][:5]) for e in meta["entry"]])
        assert torch.equal(ids // 10_000, shard)
        assert torch.equal(ids % 10_000 // 100, meta["sequence"])
        assert torch.equal(ids % 100, meta["position"])
        assert bool((meta["position"] >= burn_in).all())


def test_eval_metadata_locates_rows_in_lejepa_layout(tmp_path):
    path = make_lejepa_manifest(tmp_path / "d", d_in=8)
    source = DataSource(path)
    entries = source.paths("test")
    for rows, meta in eval_batches(source, "test", 8, 0, seed=2, with_metadata=True):
        for row, entry, sequence, position in zip(rows, meta["entry"], meta["sequence"], meta["position"]):
            offsets, lengths = source._sequence_table(entries[int(entry)])
            assert 0 <= int(position) < int(lengths[sequence])
            flat = offsets[sequence] + position
            expected = read_safetensors_rows(path.parent / entries[int(entry)], flat.reshape(1))[0]
            assert torch.equal(row, expected)


def test_moments_without_equals_recomputation():
    gen = torch.Generator().manual_seed(0)
    y = torch.randn(200, 6, generator=gen)
    full = _Moments(6, torch.device("cpu"), covariance=True)
    full.add(y)
    kept = _Moments(6, torch.device("cpu"), covariance=True)
    kept.add(y[10:])
    trimmed = full.without(y[:10])
    a, b = trimmed.summary("x/"), kept.summary("x/")
    assert a.keys() == b.keys()
    assert all(a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-12) for k in a)


def _run_diagnostics(y, gaussian, position, cfg):
    diagnostics = _OutlierDiagnostics(y.shape[1], torch.device("cpu"), cfg, ["shard-a", "shard-b"])
    full = _Moments(y.shape[1], torch.device("cpu"), covariance=True)
    reference = _Moments(y.shape[1], torch.device("cpu"), covariance=True)
    for start in range(0, len(y), 256):
        sl = slice(start, start + 256)
        meta = {"entry": torch.zeros(len(y[sl]), dtype=torch.long),
                "sequence": torch.arange(len(y[sl])), "position": position[sl]}
        full.add(y[sl])
        reference.add(gaussian[sl])
        diagnostics.add(y[sl], y[sl], y[sl], y[sl], y[sl], gaussian[sl], meta)
    out = {**full.summary("gaussian/"), **reference.summary("reference/")}
    extra, spectra = diagnostics.summary(full, reference)
    return {**out, **extra}, spectra


def test_leading_token_outliers_are_attributed_and_removed():
    """Gaussian bulk + huge first-token samples along one direction."""
    gen = torch.Generator().manual_seed(1)
    n, d = 4096, 32
    y = torch.randn(n, d, generator=gen)
    gaussian = torch.randn(n, d, generator=gen)
    position = torch.randint(1, 1024, (n,), generator=gen)
    position[:4] = 0  # 0.1% leading tokens
    direction = torch.nn.functional.normalize(torch.randn(d, generator=gen), dim=0)
    y[:4] += 300 * direction
    cfg = EvalConfig(outlier_fraction=0.001, outlier_table_size=4)
    out, spectra = _run_diagnostics(y, gaussian, position, cfg)
    assert out["outliers/count"] == 5
    assert out["outliers/position_0"] == 4 and out["outliers/leading_share"] == pytest.approx(0.8)
    assert all(row["position"] == 0 for row in out["outliers/top"])
    assert out["leading/y_sqnorm_mean"] > 1000 * out["nonleading/y_sqnorm_mean"]
    assert out["gaussian/y_sqnorm_q1"] > 100 * out["reference/y_sqnorm_q1"]
    reference_rank = out["reference/excl_leading/cov_effective_rank"]
    assert out["gaussian/excl_leading/cov_effective_rank"] == pytest.approx(reference_rank, rel=0.05)
    assert out["gaussian/trimmed/cov_effective_rank"] == pytest.approx(reference_rank, rel=0.05)
    assert spectra["full"][0] > 50 * spectra["excl_leading"][0]
    json.dumps(out)  # the table must be JSON serializable


def test_low_rank_collapse_survives_trimming():
    """Trace-matched rank-4 y: removing leading tokens / outliers must not hide it."""
    gen = torch.Generator().manual_seed(2)
    n, d, r = 4096, 32, 4
    basis = torch.linalg.qr(torch.randn(d, r, generator=gen))[0]
    y = (torch.randn(n, r, generator=gen) * (d / r) ** 0.5) @ basis.T
    gaussian = torch.randn(n, d, generator=gen)
    position = torch.randint(0, 1024, (n,), generator=gen)
    out, _ = _run_diagnostics(y, gaussian, position, EvalConfig())
    for kind in ("", "excl_leading/", "trimmed/"):
        assert out[f"gaussian/{kind}cov_effective_rank"] < r + 0.5
    assert out["reference/trimmed/cov_effective_rank"] > 0.8 * d
