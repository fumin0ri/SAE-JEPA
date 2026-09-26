from __future__ import annotations

import torch

from sae_jepa.config import ExperimentConfig
from sae_jepa.train import Trainer

from conftest import tiny_config


def _files(run):
    return sorted(p.name for p in (run / "checkpoints").iterdir())


def test_default_writes_only_latest(manifest, tmp_path):
    assert ExperimentConfig().train.keep_checkpoints is False
    run = tmp_path / "run"
    Trainer(tiny_config(manifest, run)).run()  # checkpoint_every=4, steps=8
    assert _files(run) == ["latest.pt"]
    assert torch.load(run / "checkpoints" / "latest.pt", weights_only=False)["step"] == 8


def test_keep_checkpoints_writes_identical_step_copies(manifest, tmp_path):
    run = tmp_path / "run"
    cfg = tiny_config(manifest, run)
    cfg.train.keep_checkpoints = True
    Trainer(cfg).run()
    assert _files(run) == ["latest.pt", "step-0000004.pt", "step-0000008.pt"]
    latest = (run / "checkpoints" / "latest.pt").read_bytes()
    assert (run / "checkpoints" / "step-0000008.pt").read_bytes() == latest
    assert torch.load(run / "checkpoints" / "step-0000004.pt", weights_only=False)["step"] == 4


def test_existing_step_files_are_left_alone(manifest, tmp_path):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    old = run / "checkpoints" / "step-0000002.pt"
    old.write_bytes(b"earlier run")
    Trainer(tiny_config(manifest, run)).run(max_steps=4)
    assert old.read_bytes() == b"earlier run"
    assert _files(run) == ["latest.pt", "step-0000002.pt"]


def test_changing_keep_checkpoints_does_not_block_resume(manifest, tmp_path):
    run = tmp_path / "run"
    cfg = tiny_config(manifest, run)
    cfg.train.keep_checkpoints = True
    Trainer(cfg).run(max_steps=4)
    resumed = Trainer(tiny_config(manifest, run))  # default False
    resumed.load_checkpoint(run / "checkpoints" / "latest.pt")
    resumed.run()
    assert resumed.step == 8
    assert _files(run) == ["latest.pt", "step-0000004.pt"]
