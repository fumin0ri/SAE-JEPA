"""Frozen whitening for ``model.type = whitened_residual_ae``.

``data.input_whitening_path`` may name either

* an ``sj-stage2 prepare-baselines`` front-end (``pca.pt`` / ``zca.pt``), so the
  stage-1 model starts from exactly the transform of that baseline SAE, or
* a train-fit file from ``input_whitening.fit`` (ZCA or PCA format).

Either way the result is a train-only matrix ``W`` and center ``c`` applied as
``z = (x - c) W`` to the scalar-normalized input ``x``; nothing is refitted.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from .normalization import check_normalization_matches

FORMAT = "sae-jepa-whitened-residual-input-v1"
BASELINE_FORMAT = "sae-jepa-baseline-frontend-v1"
FIT_FORMATS = {"sae-jepa-input-zca-v1": "zca", "sae-jepa-input-pca-v1": "pca"}


def _from_baseline(state: dict[str, Any], path: Path) -> dict[str, Any]:
    from .stage2 import tensor_hash  # stage2 imports this package's models

    kind = state["config"]["model"]["type"]
    if kind not in {"pca", "zca"}:
        raise ValueError(f"baseline front-end {path} is {kind!r}, not a whitening front-end")
    if state.get("sha256") != tensor_hash(state["model"]):
        raise ValueError("baseline front-end checksum mismatch")
    fit = state["baseline_provenance"]["whitening"]
    normalization = state["normalization"]
    return {"kind": kind, "matrix": state["model"]["whitening_matrix"],
            "center": state["model"]["whitening_center"],
            "mean": normalization["mean"], "scale": float(normalization["scale"]),
            "epsilon": fit["epsilon"], "count": fit["count"],
            "statistics": normalization, "source_sha256": state["sha256"],
            "convention": fit.get("convention", "")}


def _from_fit(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("split") != "train":
        raise ValueError("input whitening must be fitted on train only")
    return {"kind": FIT_FORMATS[state["format"]], "matrix": state["matrix"],
            "center": state["center"], "mean": state["mean"], "scale": float(state["scale"]),
            "epsilon": state["epsilon"], "count": state["count"], "statistics": state,
            "source_sha256": None, "convention": state.get("convention", "")}


def load_whitening(path: str | Path, source, normalization: dict[str, Any]) -> dict[str, Any]:
    """Load, validate against the run's data and normalization, and return a
    self-contained record that is stored in the checkpoint."""
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") == BASELINE_FORMAT:
        record = _from_baseline(state, path)
    elif state.get("format") in FIT_FORMATS:
        record = _from_fit(state)
    else:
        raise ValueError(f"{path} is neither a whitening baseline front-end nor a whitening fit")
    check_normalization_matches(record.pop("statistics"), source)
    if not (torch.equal(record["mean"].float(), normalization["mean"].float())
            and record["scale"] == float(normalization["scale"])):
        raise ValueError("whitening was fitted after a different scalar normalization; set "
                         "data.normalization_path to the normalization it was fitted with")
    d = source.d_in
    if record["matrix"].shape != (d, d) or record["center"].shape != (d,):
        raise ValueError("whitening shape does not match d_in")
    if not (torch.isfinite(record["matrix"]).all() and torch.isfinite(record["center"]).all()):
        raise ValueError("nonfinite whitening statistics")
    if not math.isfinite(record["epsilon"]) or record["epsilon"] <= 0 or record["count"] < 2:
        raise ValueError("invalid whitening epsilon/count")
    record.update(format=FORMAT, source_path=str(path),
                  matrix=record["matrix"].float().clone(), center=record["center"].float().clone(),
                  mean=record["mean"].float().clone())
    return record


def check_same(saved: dict[str, Any], current: dict[str, Any]) -> None:
    for key in ("matrix", "center", "mean"):
        if not torch.equal(saved[key], current[key]):
            raise ValueError("cannot resume: whitening statistics changed")
    if saved["epsilon"] != current["epsilon"] or saved["kind"] != current["kind"]:
        raise ValueError("cannot resume: whitening kind or epsilon changed")
