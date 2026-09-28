"""Frozen front-ends for stage 2.

Every front-end maps a residual ``h`` to a dense representation ``y`` and back:

    normalize(h)          x = (h - mu) / s        train mean vector, scalar scale
    encode_dense(h)       y = F(x)
    decode_normalized(y)  x_hat = D(y)            linear (affine for dense)
    denormalize(x)        h = s x + mu

    raw          y = x                                        D = identity
    pca          y = U^T x / sqrt(lambda + eps)               D = exact inverse
    dense_ae     y = G_theta(x)   (checkpoint with sigreg.weight = 0)
    dense_sigreg y = G_theta(x)   (checkpoint with sigreg.weight > 0)

Raw and PCA front-ends are stored as ``sae-jepa-frontend-v1`` files that also
record the data policy and normalization statistics, so stage 2 can check that
all candidates share data, exclusions and input scaling.  They run in float32
even under bf16 autocast: the PCA whitening gain can exceed 30x and the
dewhitening must be an exact inverse.

Probes must call the same ``encode_dense`` so that preprocessing is identical.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .data import torch_load
from .normalization import FRONTEND_FORMAT, PCA_FORMAT


class LinearFrontend(nn.Module):
    """``y = normalize(h) @ encoder_matrix`` and ``x_hat = y @ decoder_matrix``.

    Without matrices the map is the identity (the raw front-end).
    """

    def __init__(
        self,
        mean: torch.Tensor,
        scale: float | torch.Tensor,
        encoder_matrix: torch.Tensor | None = None,
        decoder_matrix: torch.Tensor | None = None,
        kind: str = "raw",
    ):
        super().__init__()
        if (encoder_matrix is None) != (decoder_matrix is None):
            raise ValueError("linear front-end needs both or neither matrices")
        self.kind = kind
        self.register_buffer("input_mean", mean.detach().float().clone())
        self.register_buffer("input_scale", torch.as_tensor(scale, dtype=torch.float32).clone())
        self.register_buffer(
            "encoder_matrix", None if encoder_matrix is None else encoder_matrix.float().clone()
        )
        self.register_buffer(
            "decoder_matrix", None if decoder_matrix is None else decoder_matrix.float().clone()
        )
        self.d_out = mean.numel() if encoder_matrix is None else encoder_matrix.shape[1]

    @classmethod
    def from_state_dict(cls, state: dict, kind: str) -> "LinearFrontend":
        return cls(state["input_mean"], state["input_scale"], state.get("encoder_matrix"),
                   state.get("decoder_matrix"), kind)

    def _fp32(self, device: torch.device):
        return torch.autocast(device_type=device.type, enabled=False)

    def normalize(self, h: torch.Tensor) -> torch.Tensor:
        with self._fp32(h.device):
            return (h.float() - self.input_mean) / self.input_scale

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        with self._fp32(x.device):
            return x.float() * self.input_scale + self.input_mean

    def encode_dense(self, h: torch.Tensor) -> torch.Tensor:
        x = self.normalize(h)
        if self.encoder_matrix is None:
            return x
        with self._fp32(h.device):
            return x @ self.encoder_matrix

    def decode_normalized(self, y: torch.Tensor) -> torch.Tensor:
        if self.decoder_matrix is None:
            return y.float()
        with self._fp32(y.device):
            return y.float() @ self.decoder_matrix


def pca_matrices(pca: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Whitening ``U diag((lambda+eps)^-1/2)`` and its exact inverse."""
    if pca.get("format") != PCA_FORMAT:
        raise ValueError("not a PCA whitening file")
    eigenvalues = pca["eigenvalues"].double().clamp_min(0) + float(pca["epsilon"])
    vectors = pca["eigenvectors"].double()
    whitening = vectors * eigenvalues.rsqrt()
    dewhitening = eigenvalues.sqrt()[:, None] * vectors.T
    return whitening.float(), dewhitening.float()


class RawFrontend(LinearFrontend):
    def __init__(self, mean: torch.Tensor, scale: float | torch.Tensor):
        super().__init__(mean, scale, kind="raw")


class PCAWhiteningFrontend(LinearFrontend):
    def __init__(self, pca: dict):
        if pca.get("format") == FRONTEND_FORMAT:
            pca = pca["pca"]
        whitening, dewhitening = pca_matrices(pca)
        super().__init__(pca["mean"], float(pca["scale"]), whitening, dewhitening, kind="pca")


def linear_frontend_from_file(state: dict) -> LinearFrontend:
    if state.get("format") != FRONTEND_FORMAT:
        raise ValueError("not a stage-2 front-end file")
    stats = state["normalization"]
    if state["kind"] == "raw":
        return RawFrontend(stats["mean"], float(stats["scale"]))
    if state["kind"] == "pca":
        pca = state["pca"]
        if not torch.equal(pca["mean"], stats["mean"]) or float(pca["scale"]) != float(stats["scale"]):
            raise ValueError("PCA was fitted with different normalization statistics")
        return PCAWhiteningFrontend(pca)
    raise ValueError(f"unknown front-end kind {state['kind']!r}")


class DenseCheckpointFrontend(nn.Module):
    def __init__(self, checkpoint: str | Path):
        super().__init__()
        from .evaluate import load_checkpoint_model

        self.model, state = load_checkpoint_model(checkpoint, torch.device("cpu"))
        self.sigreg_weight = float(state["config"]["sigreg"]["weight"])
        # Positions the encoder never saw in training; stage 2 should drop them too.
        self.skip_leading_positions = int(
            state["config"]["data"].get("skip_leading_positions", 0)
        )
        self.d_out = self.model.cfg.d_latent

    def encode_dense(self, h: torch.Tensor) -> torch.Tensor:
        return self.model.encode_dense(h)


def load_frontend(kind: str, path: str | Path | None = None) -> nn.Module:
    if kind in {"raw", "pca"}:
        state = torch_load(path)
        if state.get("format") == PCA_FORMAT and kind == "pca":
            frontend: nn.Module = PCAWhiteningFrontend(state)
        else:
            frontend = linear_frontend_from_file(state)
            if frontend.kind != kind:
                raise ValueError(f"{path} holds a {frontend.kind!r} front-end, not {kind!r}")
    elif kind in {"dense_ae", "dense_sigreg"}:
        frontend = DenseCheckpointFrontend(path)
    else:
        raise ValueError(f"unknown front-end {kind!r}")
    frontend.eval()
    for parameter in frontend.parameters():
        parameter.requires_grad_(False)
    return frontend
