from __future__ import annotations

import json

import pytest
import torch

from sae_jepa.data import DataSource, TrainBatches, eval_batches
from sae_jepa.frontends import load_frontend
from sae_jepa.normalization import compute_train_statistics, load_normalization, save_normalization
from sae_jepa.synthetic import make_lejepa_manifest, make_synthetic_manifest
from sae_jepa.train import Trainer

from conftest import tiny_config


def _sequences(manifest_path, split):
    """Usable rows of every stored sequence, read independently of DataSource."""
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    out = []
    for shard in manifest["shards"]:
        if shard["split"] != split:
            continue
        raw = (root / shard["file"]).read_bytes()
        size = int.from_bytes(raw[:8], "little")
        info = json.loads(raw[8 : 8 + size])["activations"]
        begin, end = info["data_offsets"]
        values = torch.frombuffer(bytearray(raw[8 + size + begin : 8 + size + end]), dtype=torch.int16)
        values = values.view(torch.bfloat16).reshape(info["shape"])
        for s in shard["sequences"]:
            out.append(values[s["offset"] : s["offset"] + s["length"]])
    return out


@pytest.mark.parametrize("k", [1, 3])
def test_lejepa_positions_drop_leading_tokens_of_every_sequence(tmp_path, k):
    path = make_lejepa_manifest(tmp_path / "d", d_in=8)
    source = DataSource(path, skip_leading_positions=k)
    for split in ("train", "validation", "test"):
        expected = torch.cat([seq[k:] for seq in _sequences(path, split)])
        got = torch.cat([source.positions(e) for e in source.paths(split)])
        assert torch.equal(got, expected)
        assert source.split_positions(split) == len(expected)
    entry = source.paths("validation")[0]
    rows = source.positions(entry)
    local = torch.randperm(len(rows))[:9]
    assert torch.equal(source.gather(entry, local), rows[local])


def test_eval_and_train_batches_never_contain_skipped_positions(tmp_path):
    path = make_lejepa_manifest(tmp_path / "d", d_in=8)
    source = DataSource(path, skip_leading_positions=1)
    positions = torch.cat([meta["position"] for _, meta in
                           eval_batches(source, "test", 8, 0, seed=0, with_metadata=True)])
    assert len(positions) and int(positions.min()) >= 1
    firsts = {tuple(seq[0].float().tolist()) for seq in _sequences(path, "train")}
    batches = TrainBatches(source, 16, seed=0, shards_per_window=2)
    for _ in range(10):
        assert not any(tuple(r.float().tolist()) in firsts for r in next(batches))


def test_jepa_layout_combines_with_manifest_burn_in(tmp_path):
    path = make_synthetic_manifest(tmp_path / "d", d_in=8, burn_in_tokens=2)
    assert DataSource(path).burn_in == 2
    assert DataSource(path, skip_leading_positions=1).burn_in == 2  # already excluded
    assert DataSource(path, skip_leading_positions=4).burn_in == 4
    assert DataSource(path, skip_burn_in=False, skip_leading_positions=1).burn_in == 1
    with pytest.raises(ValueError):
        DataSource(path, skip_leading_positions=-1)


def test_normalization_uses_the_same_positions_and_rejects_mismatch(tmp_path):
    path = make_lejepa_manifest(tmp_path / "d", d_in=8)
    skipped = DataSource(path, skip_leading_positions=1)
    stats = compute_train_statistics(skipped)
    expected = torch.cat([seq[1:] for seq in _sequences(path, "train")]).double()
    assert torch.allclose(stats["mean"].double(), expected.mean(0), atol=1e-4)
    assert stats["count"] == len(expected)
    save_normalization(stats, tmp_path / "norm.pt")
    load_normalization(tmp_path / "norm.pt", skipped)
    with pytest.raises(ValueError, match="skip-leading-positions"):
        load_normalization(tmp_path / "norm.pt", DataSource(path))


def test_training_with_skip_and_resume_guard(tmp_path):
    path = make_lejepa_manifest(tmp_path / "d", d_in=16)
    cfg = tiny_config(path, tmp_path / "run")
    cfg.data.skip_leading_positions = 1
    trainer = Trainer(cfg)
    assert trainer.source.burn_in == 1 and trainer.normalization["burn_in_excluded"] == 1
    final = trainer.run()
    assert final["leading/count"] == 0  # eval.leading_positions=1 finds nothing left
    assert final["outliers/position_0"] == 0
    checkpoint = tmp_path / "run" / "checkpoints" / "latest.pt"
    assert load_frontend("dense_sigreg", checkpoint).skip_leading_positions == 1

    changed = tiny_config(path, tmp_path / "run")  # default: keep position 0
    with pytest.raises(ValueError):
        Trainer(changed)  # saved normalization excluded position 0
    changed.data.normalization_path = str(tmp_path / "fresh.pt")
    save_normalization(compute_train_statistics(DataSource(path)), tmp_path / "fresh.pt")
    with pytest.raises(ValueError, match="data.skip_leading_positions"):
        Trainer(changed).load_checkpoint(checkpoint)


def test_old_checkpoints_without_the_key_still_resume(tmp_path, manifest):
    Trainer(tiny_config(manifest, tmp_path / "run")).run(max_steps=4)
    checkpoint = tmp_path / "run" / "checkpoints" / "latest.pt"
    state = torch.load(checkpoint, weights_only=False)
    del state["config"]["data"]["skip_leading_positions"]
    torch.save(state, checkpoint)
    trainer = Trainer(tiny_config(manifest, tmp_path / "run"))
    trainer.load_checkpoint(checkpoint)
    assert trainer.step == 4
