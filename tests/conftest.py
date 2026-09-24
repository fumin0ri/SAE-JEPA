from __future__ import annotations

from pathlib import Path

import pytest

from sae_jepa.config import load_config
from sae_jepa.synthetic import make_synthetic_manifest


@pytest.fixture()
def manifest(tmp_path: Path) -> Path:
    return make_synthetic_manifest(tmp_path / "data", d_in=16, sequence_length=12, train_shards=4,
                                   validation_shards=2, sequences_per_shard=8)


def tiny_config(manifest: Path, output: Path, weight: float = 0.1, steps: int = 8):
    return load_config(
        None,
        [
            f"data.activation_manifest={manifest.as_posix()}",
            "data.shards_per_window=2",
            "model.d_hidden=24",
            "model.d_latent=16",
            f"sigreg.weight={weight}",
            "sigreg.num_projections=8",
            "sigreg.validation_projections=8",
            "optim.batch_size=16",
            f"optim.steps={steps}",
            "optim.warmup_steps=2",
            "optim.lr=1e-3",
            "optim.amp_dtype=none",
            "train.device=cpu",
            f"train.output_dir={output.as_posix()}",
            "train.log_every=2",
            "train.validation_every=4",
            "train.validation_batches=2",
            "train.checkpoint_every=4",
            "eval.batch_size=16",
            "eval.batches=3",
            "eval.diagnostic_projections=8",
            "eval.maximum_quantile_samples=64",
        ],
    )
