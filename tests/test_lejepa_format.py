from __future__ import annotations

import json

import pytest
import torch

from sae_jepa.data import LEJEPA_FORMAT, DataSource, TrainBatches, eval_batches
from sae_jepa.normalization import compute_train_statistics
from sae_jepa.synthetic import make_lejepa_manifest
from sae_jepa.train import Trainer

from conftest import tiny_config


def _flat(manifest_path, split):
    """Expected usable rows of a split, read independently of DataSource."""
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    rows = []
    for shard in manifest["shards"]:
        if shard["split"] != split:
            continue
        raw = (root / shard["file"]).read_bytes()
        size = int.from_bytes(raw[:8], "little")
        header = json.loads(raw[8 : 8 + size])
        info = header["activations"]
        begin, end = info["data_offsets"]
        data = bytearray(raw[8 + size + begin : 8 + size + end])
        values = torch.frombuffer(data, dtype=torch.int16).view(torch.bfloat16).reshape(info["shape"])
        for sequence in shard["sequences"]:
            rows.append(values[sequence["offset"] : sequence["offset"] + sequence["length"]])
    return torch.cat(rows)


def test_reads_lejepa_manifest_with_document_splits(tmp_path):
    path = make_lejepa_manifest(tmp_path / "d", d_in=8)
    source = DataSource(path)
    assert source.format == LEJEPA_FORMAT and source.d_in == 8 and source.burn_in == 0
    assert all(source.splits[s] for s in ("train", "validation", "test"))
    assert all(e.startswith(f"{s}/") for s in source.splits for e in source.splits[s])
    for split in ("train", "validation", "test"):
        expected = _flat(path, split)
        got = torch.cat([source.positions(e) for e in source.paths(split)])
        assert torch.equal(got, expected)
        assert source.split_positions(split) == len(expected)


def test_matches_reference_safetensors_library(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    path = make_lejepa_manifest(tmp_path / "d", d_in=8)
    source = DataSource(path)
    entry = source.paths("train")[0]
    loaded = safetensors.load_file(str(path.parent / entry))["activations"]
    assert torch.equal(source.positions(entry), loaded)


def test_gather_and_eval_batches_agree_with_positions(tmp_path):
    path = make_lejepa_manifest(tmp_path / "d", d_in=8)
    source = DataSource(path)
    entry = source.paths("validation")[0]
    rows = source.positions(entry)
    local = torch.randperm(len(rows))[:7]
    assert torch.equal(source.gather(entry, local), rows[local])
    batches = list(eval_batches(source, "test", 8, 0, seed=1))
    assert batches and all(b.shape == (8, 8) for b in batches)
    pool = {tuple(r.float().tolist()) for r in _flat(path, "test")}
    assert all(tuple(r.float().tolist()) in pool for b in batches for r in b)


def test_train_statistics_and_batches_use_train_split_only(tmp_path):
    path = make_lejepa_manifest(tmp_path / "d", d_in=8)
    source = DataSource(path)
    stats = compute_train_statistics(source)
    expected = _flat(path, "train").double()
    assert torch.allclose(stats["mean"].double(), expected.mean(0), atol=1e-4)
    batch = next(TrainBatches(source, 16, seed=0, shards_per_window=2))
    pool = {tuple(r.float().tolist()) for r in expected.to(torch.bfloat16)}
    assert all(tuple(r.float().tolist()) in pool for r in batch)


def test_end_to_end_training_on_lejepa_layout(tmp_path):
    path = make_lejepa_manifest(tmp_path / "d", d_in=16)
    final = Trainer(tiny_config(path, tmp_path / "run")).run()
    assert final["split"] == "validation" and final["reconstruction/fvu"] == final["reconstruction/fvu"]


def test_unknown_manifest_layout_is_explained(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "something-else"}))
    with pytest.raises(ValueError, match="LeJEPA-SAE"):
        DataSource(tmp_path / "manifest.json")
