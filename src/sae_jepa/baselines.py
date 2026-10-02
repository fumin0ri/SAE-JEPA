"""Train-only Raw and full-dimensional ZCA front-ends for the shared SAE pipeline."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .data import write_json
from .input_whitening import fit as fit_zca

FORMAT = "sae-jepa-baseline-frontend-v1"
KINDS = ("raw", "zca")


class BaselineFrontend(nn.Module):
    def __init__(self, kind, normalization, whitening=None):
        super().__init__()
        if kind not in KINDS or (kind == "zca") != (whitening is not None):
            raise ValueError("invalid baseline kind or missing whitening statistics")
        mean = normalization["mean"].float()
        scale = float(normalization["scale"])
        if mean.ndim != 1 or not torch.isfinite(mean).all() or not 0 < scale < float("inf"):
            raise ValueError("invalid baseline normalization")
        self.cfg = SimpleNamespace(d_in=len(mean), d_latent=len(mean), type=kind)
        self.register_buffer("input_mean", mean.clone())
        self.register_buffer("input_scale", torch.tensor(scale))
        if whitening is not None:
            matrix, center = whitening["matrix"], whitening["center"]
            if matrix.shape != (len(mean), len(mean)) or center.shape != mean.shape:
                raise ValueError("invalid baseline whitening shape")
            if not torch.isfinite(matrix).all() or not torch.isfinite(center).all():
                raise ValueError("nonfinite baseline whitening")
            self.register_buffer("whitening_center", center.float().clone())
            self.register_buffer("whitening_matrix", matrix.float().clone())
            # Invert the saved float32 transform, in float64, rather than refit
            # a readout. Regularization changes the metric, not the dimension.
            inverse = torch.linalg.inv(matrix.double()).float()
            if not torch.isfinite(inverse).all():
                raise ValueError("nonfinite inverse whitening")
            self.register_buffer("inverse_whitening_matrix", inverse)

    def encode_dense(self, h):
        with torch.autocast(device_type=h.device.type, enabled=False):
            x = (h.float() - self.input_mean) / self.input_scale
            if self.cfg.type == "zca":
                x = (x - self.whitening_center) @ self.whitening_matrix
            return x

    def decode_normalized(self, y):
        with torch.autocast(device_type=y.device.type, enabled=False):
            if self.cfg.type == "zca":
                return y.float() @ self.inverse_whitening_matrix + self.whitening_center
            return y.float()

    def denormalize(self, x):
        return x.float() * self.input_scale + self.input_mean


def build_baseline(front):
    if front.get("format") != FORMAT:
        raise ValueError("unsupported baseline front-end format")
    kind = front["config"]["model"]["type"]
    state = front["model"]
    whitening = ({"matrix": state["whitening_matrix"], "center": state["whitening_center"]}
                 if kind == "zca" else None)
    model = BaselineFrontend(kind, front["normalization"], whitening)
    for key in ("input_mean", "input_scale"):
        if not torch.equal(state[key], model.state_dict()[key]):
            raise ValueError("baseline model and normalization differ")
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError("nonfinite baseline state")
    return model


def prepare(args):
    # Local import avoids a cycle: stage2 loads this module to dispatch baselines.
    from .stage2 import load_front, source_for, tensor_hash

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("baseline output must be empty; choose a new directory")
    if len(set(args.kinds)) != len(args.kinds):
        raise ValueError("duplicate baseline kinds")
    reference, front = load_front(args.reference_checkpoint)
    source = source_for(front, args.activation_manifest)
    if reference.cfg.d_latent != source.d_in:
        raise ValueError("full-dimensional baselines require a reference with d_latent == d_in")
    del reference
    normalization = deepcopy(front["normalization"])
    # Older Stage1 checkpoints omit the split label; source_for has already
    # verified the recorded normalization shards equal precisely the train set.
    if normalization.get("split", "train") != "train":
        raise ValueError("baseline normalization must be train-only")
    normalization["split"] = "train"
    whitening = None
    if "zca" in args.kinds:
        print("Fitting full-dimensional ZCA on train activations only", flush=True)
        whitening = fit_zca(source, normalization, epsilon=args.epsilon,
            maximum_positions=args.maximum_positions, chunk_size=args.chunk_size,
            sample_seed=args.sample_seed, device=args.device)
    output.mkdir(parents=True, exist_ok=True)
    for kind in args.kinds:
        model = BaselineFrontend(kind, normalization, whitening if kind == "zca" else None)
        # Only data policy is inherited; no learned encoder, readout, or input
        # whitening from the reference is part of either baseline.
        cfg = {"name": "Raw" if kind == "raw" else "ZCA whitening",
               "model": {"type": kind, "d_in": source.d_in, "d_latent": source.d_in},
               "data": deepcopy(front["config"]["data"]), "sigreg": {"weight": 0.0}}
        cfg["data"]["activation_manifest"] = str(source.manifest_path)
        cfg["data"]["input_whitening_path"] = None
        provenance = {"reference_checkpoint": str(args.reference_checkpoint),
                      "reference_sha256": front["sha256"],
                      "kind": kind, "dimension_reduction": False}
        if kind == "zca":
            provenance["whitening"] = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in whitening.items() if k not in {"mean", "center", "matrix"}}
        state = {"format": FORMAT, "config": cfg, "normalization": normalization,
                 "data_manifest": source.record(), "model": model.state_dict(),
                 "step": 0, "baseline_provenance": provenance}
        state["sha256"] = tensor_hash(state["model"])
        path = output / f"{kind}.pt"
        torch.save(state, path)
        write_json(output / f"{kind}.json", {"config": cfg, "sha256": state["sha256"],
                   "data_manifest": source.record(), "provenance": provenance})
        print(f"Saved {kind} front-end: {path}", flush=True)
