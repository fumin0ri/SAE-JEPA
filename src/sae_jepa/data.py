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


def parse_entry(entry: str) -> tuple[str, int, int | None]:
    """Split entry ``"path"`` or ``"path#rows=start:stop"`` (sequence rows)."""
    path, marker, rows = entry.partition("#rows=")
    if not marker:
        return path, 0, None
    start, stop = rows.split(":")
    return path, int(start), int(stop)


def format_entry(path: str, start: int, stop: int) -> str:
    return f"{path}#rows={start}:{stop}"


def resolve_splits(
    manifest: dict[str, Any],
    test_split: str = "auto",
    holdout_test_fraction: float = 0.5,
    root: Path | None = None,
) -> dict[str, list[str]]:
    """Return split entries for train / validation / test.

    ``test_split``:
      * ``"manifest"`` -- use ``manifest["test"]`` (error if absent);
      * ``"holdout"`` -- reserve the trailing ``holdout_test_fraction`` of the
        validation shards as test shards.  With a single validation shard the
        split is made at the sequence level inside that shard (entries of the
        form ``path#rows=start:stop``);
      * ``"auto"`` -- ``manifest`` when present, otherwise ``holdout``;
      * ``"none"`` -- no test split.
    A holdout that cannot produce non-empty validation and test splits raises
    here, before any training, instead of failing at test evaluation.
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
        if len(validation) >= 2:
            n_test = round(len(validation) * holdout_test_fraction)
            n_test = min(len(validation) - 1, max(1, n_test))
            test = validation[len(validation) - n_test :]
            validation = validation[: len(validation) - n_test]
        else:
            if root is None:
                raise ValueError("a sequence-level holdout needs the manifest root")
            (only,) = validation
            sequences = len(load_valid_lengths(root / only))
            if sequences < 2:
                raise ValueError(
                    "cannot hold out a test split: the only validation shard has "
                    f"{sequences} sequence(s); extract more validation data or set "
                    "data.test_split=none and evaluate on validation only"
                )
            n_test = min(sequences - 1, max(1, round(sequences * holdout_test_fraction)))
            validation = [format_entry(only, 0, sequences - n_test)]
            test = [format_entry(only, sequences - n_test, sequences)]
    splits = {"train": train, "validation": validation, "test": test}
    assert_disjoint_splits(splits)
    return splits


def assert_disjoint_splits(splits: dict[str, list[str]]) -> None:
    claimed: dict[str, list[tuple[str, int, float]]] = {}
    for split, entries in splits.items():
        for entry in entries:
            path, start, stop = parse_entry(entry)
            key = Path(path).as_posix()
            end = math.inf if stop is None else stop
            for other_split, other_start, other_end in claimed.get(key, []):
                if start < other_end and other_start < end:
                    raise ValueError(f"{entry} overlaps data already assigned to {other_split}")
            claimed.setdefault(key, []).append((split, start, end))


def _load(path: Path, mmap: bool) -> Any:
    if mmap:
        try:
            return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except (RuntimeError, ValueError):
            pass  # legacy (non-zip) serialization cannot be memory-mapped
    return torch_load(path)


def load_valid_lengths(path: Path) -> torch.Tensor:
    return _load(path, mmap=True)["valid_lengths"].long().clone()


def load_sequence_shard(
    path: Path, sequence_length: int, mmap: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    value = _load(path, mmap)
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
    return activations[_position_mask(valid_lengths, sequence_length, burn_in)]


def _position_mask(valid_lengths: torch.Tensor, sequence_length: int, burn_in: int) -> torch.Tensor:
    positions = torch.arange(sequence_length)
    return (positions[None, :] >= burn_in) & (positions[None, :] < valid_lengths[:, None])


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
        self.splits = resolve_splits(
            self.manifest, test_split, holdout_test_fraction, self.root
        )
        self.fingerprint = manifest_fingerprint(self.manifest)
        self._counts: dict[str, torch.Tensor] = {}

    def paths(self, split: str) -> list[str]:
        """Split entries: shard paths, optionally with a sequence-row range."""
        return list(self.splits[split])

    def _resolve(self, entry: str) -> tuple[Path, int, int | None]:
        path, start, stop = parse_entry(entry)
        return self.root / path, start, stop

    def positions(self, entry: str) -> torch.Tensor:
        """All valid positions of an entry as ``[n, d]`` in storage order."""
        path, start, stop = self._resolve(entry)
        activations, valid_lengths = load_sequence_shard(path, self.sequence_length)
        activations, valid_lengths = activations[start:stop], valid_lengths[start:stop]
        return activations[_position_mask(valid_lengths, self.sequence_length, self.burn_in)]

    def sequence_counts(self, entry: str) -> torch.Tensor:
        """Usable positions per sequence of an entry (cached; memory-mapped read)."""
        if entry not in self._counts:
            path, start, stop = self._resolve(entry)
            lengths = load_valid_lengths(path)[start:stop]
            self._counts[entry] = (lengths - self.burn_in).clamp_min(0)
        return self._counts[entry]

    def split_positions(self, split: str) -> int:
        return sum(int(self.sequence_counts(entry).sum()) for entry in self.paths(split))

    def gather(self, entry: str, local: torch.Tensor) -> torch.Tensor:
        """Rows ``local`` (indices in :meth:`positions` order) read via a memory map."""
        path, start, _ = self._resolve(entry)
        counts = self.sequence_counts(entry)
        ends = counts.cumsum(0)
        sequence = torch.searchsorted(ends, local, right=True)
        position = self.burn_in + local - (ends - counts)[sequence]
        activations, _ = load_sequence_shard(path, self.sequence_length, mmap=True)
        return activations[sequence + start, position].clone()

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
    source: DataSource,
    split: str,
    batch_size: int,
    maximum_batches: int = 0,
    seed: int = 0,
    chunk_batches: int = 16,
) -> Iterator[torch.Tensor]:
    """Fixed, shuffled evaluation batches drawn from the whole held-out split.

    Consecutive tokens of one document are strongly correlated, so storage-order
    batches make even a per-token Gaussian look non-Gaussian to the batch-level
    SIGReg statistic.  Instead a generator seeded only by ``seed`` draws
    ``maximum_batches * batch_size`` positions without replacement from all
    sequences and shards of the split (every full batch when ``maximum_batches``
    is 0) and assigns them to batches in random order.  The sample depends only
    on (data, split, batch_size, maximum_batches, seed), so every model and every
    lambda sees identical batches.  All batches have the same ``N`` because the
    N-scaled SIGReg statistic is only comparable at a fixed batch size.  Rows
    are read through memory maps in chunks of ``chunk_batches`` batches.
    """
    if split == "train":
        raise ValueError("evaluation batches are only drawn from held-out splits")
    entries = source.paths(split)
    if not entries:
        raise ValueError(f"split {split!r} has no shards")
    counts = torch.tensor([int(source.sequence_counts(entry).sum()) for entry in entries])
    total = int(counts.sum())
    n_batches = total // batch_size
    if maximum_batches > 0:
        n_batches = min(n_batches, maximum_batches)
    if n_batches == 0:
        return
    order = torch.randperm(total, generator=torch.Generator().manual_seed(seed))
    order = order[: n_batches * batch_size]
    ends = counts.cumsum(0)
    starts = ends - counts
    step = max(1, chunk_batches) * batch_size
    for chunk_start in range(0, len(order), step):
        slots = order[chunk_start : chunk_start + step]
        owner = torch.searchsorted(ends, slots, right=True)
        buffer: torch.Tensor | None = None
        for index in torch.unique(owner).tolist():
            mask = owner == index
            rows = source.gather(entries[index], slots[mask] - starts[index])
            if buffer is None:
                buffer = torch.empty(len(slots), rows.shape[-1], dtype=rows.dtype)
            buffer[mask] = rows
        assert buffer is not None
        for offset in range(0, len(slots), batch_size):
            yield buffer[offset : offset + batch_size]
