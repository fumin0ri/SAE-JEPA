"""Small synthetic activation manifests (tests and smoke runs)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import ACTIVATION_FORMAT, write_json


def make_synthetic_manifest(
    output_dir: str | Path,
    *,
    d_in: int = 32,
    sequence_length: int = 16,
    burn_in_tokens: int = 2,
    train_shards: int = 4,
    validation_shards: int = 2,
    sequences_per_shard: int = 16,
    seed: int = 0,
    validation_offset: float = 0.0,
) -> Path:
    """Correlated, anisotropic, non-Gaussian residual-like data.

    ``validation_offset`` shifts only the held-out split; tests use it to check
    that train statistics never see validation data.
    """
    output = Path(output_dir)
    generator = torch.Generator().manual_seed(seed)
    latent_dim = max(2, d_in // 2)
    mixing = torch.randn(latent_dim, d_in, generator=generator) / latent_dim**0.5
    mixing[0] *= 4.0  # one dominant direction, as in real residual streams
    bias = torch.randn(d_in, generator=generator) * 3.0
    manifest = {
        "format": ACTIVATION_FORMAT,
        "dataset": {"name": "synthetic"},
        "model": "synthetic",
        "layer": 0,
        "hook_point": "post",
        "sequence_length": sequence_length,
        "burn_in_tokens": burn_in_tokens,
        "d_in": d_in,
        "seed": seed,
    }
    for split, count in (("train", train_shards), ("validation", validation_shards)):
        shards = []
        positions = 0
        for index in range(count):
            latent = torch.randn(sequences_per_shard, sequence_length, latent_dim, generator=generator)
            latent = latent.sign() * latent.abs().pow(1.5)  # heavy tails
            noise = 0.1 * torch.randn(sequences_per_shard, sequence_length, d_in, generator=generator)
            activations = latent @ mixing + bias + noise  # full-rank covariance
            if split == "validation":
                activations = activations + validation_offset
            lengths = torch.randint(
                burn_in_tokens + 2, sequence_length + 1, (sequences_per_shard,), generator=generator
            )
            relative = f"{split}/shard-{index:05d}.pt"
            (output / split).mkdir(parents=True, exist_ok=True)
            torch.save(
                {"activations": activations.to(torch.bfloat16), "valid_lengths": lengths.int()},
                output / relative,
            )
            shards.append(relative)
            positions += int(lengths.sum())
        manifest[split] = {"shards": shards, "sequences": count * sequences_per_shard, "positions": positions}
    write_json(output / "manifest.json", manifest)
    return output / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a synthetic activation manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--d-in", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--train-shards", type=int, default=8)
    parser.add_argument("--validation-shards", type=int, default=4)
    parser.add_argument("--sequences-per-shard", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = make_synthetic_manifest(
        args.output_dir,
        d_in=args.d_in,
        sequence_length=args.sequence_length,
        train_shards=args.train_shards,
        validation_shards=args.validation_shards,
        sequences_per_shard=args.sequences_per_shard,
        seed=args.seed,
    )
    print(path)


if __name__ == "__main__":
    main()
