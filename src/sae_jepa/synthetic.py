"""Small synthetic activation manifests (tests and smoke runs)."""

from __future__ import annotations

import argparse
import json
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


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Minimal safetensors writer (keeps the package free of that dependency)."""
    names = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.int32: "I32"}
    header: dict[str, dict] = {}
    payload = []
    offset = 0
    for name, tensor in tensors.items():
        tensor = tensor.contiguous()
        raw = (tensor.view(torch.int16) if tensor.dtype == torch.bfloat16 else tensor).numpy().tobytes()
        header[name] = {
            "dtype": names[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        payload.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(len(encoded).to_bytes(8, "little"))
        f.write(encoded)
        for raw in payload:
            f.write(raw)


def make_lejepa_manifest(
    output_dir: str | Path,
    *,
    d_in: int = 32,
    context_length: int = 16,
    documents: dict[str, int] | None = None,
    shard_tokens: int = 64,
    seed: int = 0,
) -> Path:
    """Synthetic activations in LeJEPA-SAE ``extract`` layout (flat safetensors shards)."""
    output = Path(output_dir)
    documents = documents or {"train": 24, "validation": 8, "test": 8}
    generator = torch.Generator().manual_seed(seed)
    mixing = torch.randn(d_in, d_in, generator=generator) / d_in**0.5
    mixing[0] *= 4.0
    bias = torch.randn(d_in, generator=generator) * 3.0
    shards: list[dict] = []
    for split, count in documents.items():
        pending: list[tuple[torch.Tensor, str, int]] = []
        pending_tokens = 0

        def flush() -> None:
            nonlocal pending, pending_tokens
            if not pending:
                return
            relative = f"{split}/shard-{sum(s['split'] == split for s in shards):05d}.safetensors"
            activations = torch.cat([a for a, _, _ in pending]).to(torch.bfloat16)
            tokens = torch.zeros(len(activations), dtype=torch.int32)
            write_safetensors(output / relative, {"activations": activations, "token_ids": tokens})
            sequences, offset = [], 0
            for values, document, segment in pending:
                sequences.append({"offset": offset, "length": len(values),
                                  "document_id": document, "segment_index": segment})
                offset += len(values)
            shards.append({"file": relative, "split": split, "num_tokens": offset,
                           "sequences": sequences})
            pending, pending_tokens = [], 0

        for index in range(count):
            length = int(torch.randint(2, 2 * context_length, (1,), generator=generator))
            latent = torch.randn(length, d_in, generator=generator)
            values = (latent.sign() * latent.abs().pow(1.5)) @ mixing + bias
            for segment, start in enumerate(range(0, length, context_length)):
                pending.append((values[start : start + context_length], f"{split}-{index}", segment))
                pending_tokens += len(pending[-1][0])
                if pending_tokens >= shard_tokens:
                    flush()
        flush()
    manifest = {
        "format_version": 1,
        "model": "synthetic",
        "revision": None,
        "hook_point": "block_output:0",
        "layer": 0,
        "d_llm": d_in,
        "dtype": "bfloat16",
        "context_length": context_length,
        "minimum_window_size": 1,
        "dataset": "synthetic",
        "split_unit": "document",
        "split_seed": seed,
        "tokens_by_split": {
            split: sum(s["num_tokens"] for s in shards if s["split"] == split) for split in documents
        },
        "shards": shards,
    }
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
    parser.add_argument("--layout", choices=["jepa-sae", "lejepa-sae"], default="jepa-sae")
    args = parser.parse_args()
    if args.layout == "lejepa-sae":
        sequences = args.sequences_per_shard
        path = make_lejepa_manifest(
            args.output_dir,
            d_in=args.d_in,
            context_length=args.sequence_length,
            documents={
                "train": args.train_shards * sequences,
                "validation": args.validation_shards * sequences // 2,
                "test": args.validation_shards * sequences // 2,
            },
            shard_tokens=sequences * args.sequence_length,
            seed=args.seed,
        )
        print(path)
        return
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
