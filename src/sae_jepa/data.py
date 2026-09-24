"""Reading residual activations produced by JEPA-SAE's ``sr-extract-pile``.

The on-disk format is ``shared-residual-sequence-shards-v2``: each shard is a
``torch.save`` dict with ``activations`` of shape ``[n, sequence_length, d_in]``
and ``valid_lengths`` of shape ``[n]``.  Stage 1 treats every valid token
position (optionally after the burn-in prefix) as one independent sample.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator

import torch


ACTIVATION_FORMAT = "shared-residual-sequence-shards-v2"
SPLITS = ("train", "validation", "test")


def torch_load(path: str | Path) -> Any:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def write_json(path: str | Path, value: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def mix_seed(*values: int) -> int:
    """Deterministically combine integers into a 63-bit generator seed."""
    encoded = ",".join(str(int(value)) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") >> 1


def load_activation_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != ACTIVATION_FORMAT:
        raise ValueError(f"unsupported activation manifest format at {manifest_path}")
    for key in ("d_in", "sequence_length", "train", "validation"):
        if key not in manifest:
            raise ValueError(f"activation manifest is missing {key!r}")
    for split in ("train", "validation"):
        if not manifest[split].get("shards"):
            raise ValueError(f"activation manifest has no {split} shards")
    return manifest_path.parent, manifest


def manifest_fingerprint(manifest: dict[str, Any]) -> str:
    identity = {
        key: manifest.get(key)
        for key in (
            "format",
            "dataset",
            "model",
            "resolved_model_revision",
            "layer",
            "layer_path",
            "hook_point",
            "sequence_length",
            "burn_in_tokens",
            "d_in",
            "seed",
        )
    }
    for split in SPLITS:
        if split in manifest:
            identity[split] = {
                key: manifest[split].get(key)
                for key in ("sequences", "positions", "shards")
            }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_splits(
    manifest: dict[str, Any], test_split: str = "auto", holdout_test_fraction: float = 0.5
) -> dict[str, list[str]]:
    """Return relative shard paths for train / validation / test.

    ``test_split``:
      * ``"manifest"`` -- use ``manifest["test"]`` (error if absent);
      * ``"holdout"`` -- reserve the trailing ``holdout_test_fraction`` of the
        validation shards as test shards;
      * ``"auto"`` -- ``manifest`` when present, otherwise ``holdout``;
      * ``"none"`` -- no test split.
    Train shards are never modified, so normalization statistics computed from
    the train split cannot depend on validation or test data.
    """
    if test_split not in {"auto", "manifest", "holdout", "none"}:
        raise ValueError(f"unknown test_split mode {test_split!r}")
    train = list(manifest["train"]["shards"])
    validation = list(manifest["validation"]["shards"])
    test: list[str] = []
    has_manifest_test = bool(manifest.get("test", {}).get("shards"))
    mode = test_split
    if mode == "auto":
        mode = "manifest" if has_manifest_test else "holdout"
    if mode == "manifest":
        if not has_manifest_test:
            raise ValueError("test_split='manifest' but the manifest has no test shards")
        test = list(manifest["test"]["shards"])
    elif mode == "holdout":
        if not 0.0 < holdout_test_fraction < 1.0:
            raise ValueError("holdout_test_fraction must lie in (0, 1)")
        n_test = math.floor(len(validation) * holdout_test_fraction)
        if n_test >= 1 and len(validation) - n_test >= 1:
            test = validation[len(validation) - n_test :]
            validation = validation[: len(validation) - n_test]
    splits = {"train": train, "validation": validation, "test": test}
    assert_disjoint_splits(splits)
    return splits


def assert_disjoint_splits(splits: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for split, paths in splits.items():
        for path in paths:
            key = str(Path(path).as_posix())
            if key in seen and seen[key] != split:
                raise ValueError(f"shard {path} appears in both {seen[key]} and {split}")
            seen[key] = split


def load_sequence_shard(path: Path, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch_load(path)
    if not isinstance(value, dict) or "activations" not in value:
        raise ValueError(f"invalid sequence shard at {path}")
    activations = value["activations"]
    valid_lengths = value.get("valid_lengths")
    if activations.ndim != 3 or activations.shape[1] != sequence_length:
        raise ValueError(f"invalid activation shard shape at {path}")
    if valid_lengths is None or valid_lengths.shape != (len(activations),):
        raise ValueError(f"invalid valid_lengths at {path}")
    return activations, valid_lengths.long()


def shard_positions(
    path: Path, sequence_length: int, burn_in: int
) -> torch.Tensor:
    """All valid positions ``[burn_in, valid_length)`` of a shard as ``[n, d]``."""
    activations, valid_lengths = load_sequence_shard(path, sequence_length)
    positions = torch.arange(sequence_length)
    mask = (positions[None, :] >= burn_in) & (positions[None, :] < valid_lengths[:, None])
    return activations[mask]


class DataSource:
    """Resolved view of a manifest: splits, burn-in, and shard loading."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        skip_burn_in: bool = True,
        test_split: str = "auto",
        holdout_test_fraction: float = 0.5,
    ):
        self.manifest_path = Path(manifest_path)
        self.root, self.manifest = load_activation_manifest(manifest_path)
        self.sequence_length = int(self.manifest["sequence_length"])
        self.d_in = int(self.manifest["d_in"])
        self.burn_in = int(self.manifest.get("burn_in_tokens", 0)) if skip_burn_in else 0
        self.splits = resolve_splits(self.manifest, test_split, holdout_test_fraction)
        self.fingerprint = manifest_fingerprint(self.manifest)

    def paths(self, split: str) -> list[Path]:
        return [self.root / relative for relative in self.splits[split]]

    def positions(self, path: Path) -> torch.Tensor:
        return shard_positions(path, self.sequence_length, self.burn_in)

    def record(self) -> dict[str, Any]:
        """Data manifest record stored with checkpoints and reports."""
        return {
            "activation_manifest": str(self.manifest_path),
            "fingerprint": self.fingerprint,
            "format": self.manifest.get("format"),
            "model": self.manifest.get("model"),
            "resolved_model_revision": self.manifest.get("resolved_model_revision"),
            "layer": self.manifest.get("layer"),
            "hook_point": self.manifest.get("hook_point"),
            "d_in": self.d_in,
            "sequence_length": self.sequence_length,
            "burn_in_excluded": self.burn_in,
            "splits": {split: list(paths) for split, paths in self.splits.items()},
        }


class TrainBatches:
    """Infinite, deterministic, resumable batches of train positions.

    Each epoch permutes the shard order with a generator seeded by
    ``(seed, epoch)``.  Consecutive groups of ``shards_per_window`` shards form a
    window whose positions are permuted by ``(seed, epoch, window)``.  Because
    every permutation is a pure function of these integers, the iterator state
    is just ``(epoch, window, offset)`` and resuming never needs buffered data.
    Positions left over at the end of a window (< one batch) are dropped.
    """

    def __init__(
        self,
        source: DataSource,
        batch_size: int,
        seed: int,
        shards_per_window: int = 4,
    ):
        if batch_size < 1 or shards_per_window < 1:
            raise ValueError("batch_size and shards_per_window must be positive")
        self.source = source
        self.paths = source.paths("train")
        self.batch_size = batch_size
        self.seed = seed
        self.shards_per_window = shards_per_window
        self.epoch = 0
        self.window = 0
        self.offset = 0
        self._rows: torch.Tensor | None = None
        self._order: torch.Tensor | None = None

    @property
    def windows_per_epoch(self) -> int:
        return math.ceil(len(self.paths) / self.shards_per_window)

    def __iter__(self) -> "TrainBatches":
        return self

    def _load_window(self) -> None:
        shard_order = torch.randperm(
            len(self.paths),
            generator=torch.Generator().manual_seed(mix_seed(self.seed, self.epoch)),
        ).tolist()
        start = self.window * self.shards_per_window
        selected = shard_order[start : start + self.shards_per_window]
        self._rows = torch.cat([self.source.positions(self.paths[i]) for i in selected])
        self._order = torch.randperm(
            len(self._rows),
            generator=torch.Generator().manual_seed(
                mix_seed(self.seed, self.epoch, self.window)
            ),
        )

    def _advance_window(self) -> None:
        self._rows = None
        self._order = None
        self.offset = 0
        self.window += 1
        if self.window >= self.windows_per_epoch:
            self.window = 0
            self.epoch += 1

    def __next__(self) -> torch.Tensor:
        empty_windows = 0
        while True:
            if self._rows is None:
                self._load_window()
            assert self._rows is not None and self._order is not None
            if self.offset + self.batch_size <= len(self._rows):
                index = self._order[self.offset : self.offset + self.batch_size]
                self.offset += self.batch_size
                return self._rows.index_select(0, index)
            if self.offset == 0:
                empty_windows += 1
                if empty_windows > self.windows_per_epoch:
                    raise RuntimeError(
                        "no train window holds a full batch; increase "
                        "shards_per_window or reduce batch_size"
                    )
            self._advance_window()

    def state_dict(self) -> dict[str, int]:
        return {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "shards_per_window": self.shards_per_window,
            "epoch": self.epoch,
            "window": self.window,
            "offset": self.offset,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        for key in ("seed", "batch_size", "shards_per_window"):
            if int(state[key]) != getattr(self, key):
                raise ValueError(f"cannot resume data iterator: {key} changed")
        self.epoch = int(state["epoch"])
        self.window = int(state["window"])
        self.offset = int(state["offset"])
        self._rows = None
        self._order = None


def eval_batches(
    source: DataSource, split: str, batch_size: int, maximum_batches: int = 0
) -> Iterator[torch.Tensor]:
    """Deterministic full batches of held-out positions in storage order.

    A trailing partial batch is dropped so that every batch has the same ``N``;
    the N-scaled SIGReg statistic is only comparable at a fixed batch size.
    """
    if split == "train":
        raise ValueError("evaluation batches are only drawn from held-out splits")
    if not source.splits.get(split):
        raise ValueError(f"split {split!r} has no shards")
    emitted = 0
    pending: torch.Tensor | None = None
    for path in source.paths(split):
        rows = source.positions(path)
        if pending is not None and len(pending):
            rows = torch.cat([pending, rows])
        start = 0
        while start + batch_size <= len(rows):
            yield rows[start : start + batch_size]
            emitted += 1
            start += batch_size
            if maximum_batches > 0 and emitted >= maximum_batches:
                return
        pending = rows[start:]
