from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sae_jepa.data import ACTIVATION_FORMAT, DataSource, eval_batches, write_json
from sae_jepa.sigreg import epps_pulley_sigreg, integration_grid, sample_projections
from sae_jepa.train import Trainer
from sae_jepa.synthetic import make_synthetic_manifest

from conftest import tiny_config


def write_manifest(root: Path, splits: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
                   sequence_length: int, burn_in: int) -> Path:
    manifest = {"format": ACTIVATION_FORMAT, "model": "custom", "sequence_length": sequence_length,
                "burn_in_tokens": burn_in}
    for split, shards in splits.items():
        entries = []
        for index, (activations, lengths) in enumerate(shards):
            relative = f"{split}/shard-{index:05d}.pt"
            (root / split).mkdir(parents=True, exist_ok=True)
            torch.save({"activations": activations, "valid_lengths": lengths}, root / relative)
            entries.append(relative)
        manifest[split] = {"shards": entries, "sequences": sum(len(a) for a, _ in shards)}
        manifest["d_in"] = shards[0][0].shape[-1]
    write_json(root / "manifest.json", manifest)
    return root / "manifest.json"


def id_shard(shard: int, sequences: int, length: int, lengths: torch.Tensor) -> tuple:
    """Feature 0 holds a unique id per (shard, sequence, position)."""
    ids = (shard * 10_000 + torch.arange(sequences)[:, None] * 100 + torch.arange(length)[None, :]).float()
    activations = torch.randn(sequences, length, 4)
    activations[..., 0] = ids
    return activations, lengths


def test_eval_batches_mix_whole_split_and_are_fixed(tmp_path):
    length, burn_in = 20, 3
    gen = torch.Generator().manual_seed(0)
    shards = [id_shard(s, 6, length, torch.randint(burn_in + 1, length + 1, (6,), generator=gen))
              for s in range(3)]
    manifest = write_manifest(tmp_path, {"train": shards[:1], "validation": shards}, length, burn_in)
    source = DataSource(manifest, test_split="none")
    valid_ids = {int(v) for entry in source.paths("validation") for v in source.positions(entry)[:, 0]}
    total = len(valid_ids)
    assert total == source.split_positions("validation")

    batches = list(eval_batches(source, "validation", 8, 0, seed=5, chunk_batches=2))
    ids = torch.cat(batches)[:, 0].long().tolist()
    assert len(batches) == total // 8 and all(len(b) == 8 for b in batches)
    assert len(set(ids)) == len(ids) and set(ids) <= valid_ids
    # batches mix shards instead of following storage order
    assert any(len({i // 10_000 for i in b[:, 0].long().tolist()}) > 1 for b in batches)
    assert ids != sorted(ids)
    # identical across calls (shared by every lambda) and depends on the seed
    again = torch.cat(list(eval_batches(source, "validation", 8, 0, seed=5)))
    assert torch.equal(again, torch.cat(batches))
    other = torch.cat(list(eval_batches(source, "validation", 8, 0, seed=6)))
    assert not torch.equal(other, again)
    # a capped evaluation is a sample of the whole split, not its first shard
    capped = torch.cat(list(eval_batches(source, "validation", 8, 3, seed=5)))[:, 0].long()
    assert len({int(i) // 10_000 for i in capped}) > 1


def test_shuffled_eval_removes_temporal_correlation_bias(tmp_path):
    """Per-token N(0, I) with strong within-sequence correlation."""
    gen = torch.Generator().manual_seed(1)
    d, length, rho = 8, 64, 0.98

    def ar_shard(sequences: int):
        x = torch.empty(sequences, length, d)
        x[:, 0] = torch.randn(sequences, d, generator=gen)
        for t in range(1, length):
            x[:, t] = rho * x[:, t - 1] + (1 - rho**2) ** 0.5 * torch.randn(sequences, d, generator=gen)
        return x, torch.full((sequences,), length)

    shards = [ar_shard(32) for _ in range(4)]
    manifest = write_manifest(tmp_path, {"train": shards[:1], "validation": shards}, length, 0)
    source = DataSource(manifest, test_split="none")
    t, w = integration_grid()
    projections = sample_projections(d, 64, torch.Generator().manual_seed(2))

    def mean_sigreg(batches):
        values = [float(epps_pulley_sigreg(b.float(), projections, t, w)) for b in batches]
        return sum(values) / len(values)

    storage = torch.cat([source.positions(e) for e in source.paths("validation")])
    storage_order = mean_sigreg(storage[: len(storage) // 256 * 256].split(256))
    shuffled = mean_sigreg(list(eval_batches(source, "validation", 256, 0, seed=3)))
    reference = mean_sigreg([torch.randn(256, d, generator=gen) for _ in range(16)])
    assert storage_order > 5 * reference
    assert shuffled < 2 * reference


def test_single_validation_shard_is_split_by_sequence(tmp_path):
    manifest = make_synthetic_manifest(tmp_path / "d", d_in=8, validation_shards=1, sequences_per_shard=10)
    source = DataSource(manifest)
    (validation,), (test,) = source.splits["validation"], source.splits["test"]
    assert validation.endswith("#rows=0:5") and test.endswith("#rows=5:10")
    full = torch.cat([source.positions(e) for e in source.paths("validation")] +
                     [source.positions(e) for e in source.paths("test")])
    assert torch.equal(full, source.positions(validation.split("#")[0]))
    assert source.split_positions("validation") + source.split_positions("test") == len(full)
    assert len(list(eval_batches(source, "test", 4, 2))) == 2


def test_unsplittable_validation_fails_before_training(tmp_path):
    manifest = make_synthetic_manifest(tmp_path / "d", d_in=8, validation_shards=1, sequences_per_shard=1)
    with pytest.raises(ValueError, match="cannot hold out a test split"):
        DataSource(manifest)
    cfg = tiny_config(manifest, tmp_path / "run")
    cfg.data.test_split = "none"
    cfg.eval.batch_size = 2
    with pytest.raises(ValueError, match="required split 'test' is empty"):
        Trainer(cfg)
    cfg.eval.required_splits = ("validation",)
    Trainer(cfg)  # validation-only evaluation is allowed when requested explicitly
    cfg.eval.batch_size = 10_000
    with pytest.raises(ValueError, match="fewer than one evaluation batch"):
        Trainer(cfg)


@pytest.mark.parametrize(
    "override",
    [("optim", "weight_decay", 0.1), ("sigreg", "scale_by_batch_size", False),
     ("sigreg", "resample_every_step", False), ("optim", "betas", (0.9, 0.95)),
     ("data", "shards_per_window", 1), ("optim", "gradient_clip", 1.0)],
)
def test_resume_rejects_changed_training_settings(manifest, tmp_path, override):
    Trainer(tiny_config(manifest, tmp_path / "run")).run(max_steps=4)
    cfg = tiny_config(manifest, tmp_path / "run")
    section, key, value = override
    setattr(getattr(cfg, section), key, value)
    trainer = Trainer(cfg)
    with pytest.raises(ValueError, match=f"{section}.{key}"):
        trainer.load_checkpoint(tmp_path / "run" / "checkpoints" / "latest.pt")


def test_resume_allows_logging_and_eval_changes(manifest, tmp_path):
    Trainer(tiny_config(manifest, tmp_path / "run")).run(max_steps=4)
    cfg = tiny_config(manifest, tmp_path / "run")
    cfg.train.log_every = 1
    cfg.train.validation_every = 2
    cfg.eval.batches = 1
    trainer = Trainer(cfg)
    trainer.load_checkpoint(tmp_path / "run" / "checkpoints" / "latest.pt")
    assert trainer.step == 4
    assert trainer.optimizer.param_groups[0]["weight_decay"] == cfg.optim.weight_decay
