"""Frozen front-ends for stage 2: every one exposes ``encode_dense(h)``.

Stage 2 attaches the same Top-K SAE to each front-end:

    raw          y = h
    pca          y = U^T ((h - mu)/s) / sqrt(lambda + eps)
    dense_ae     y = G_theta((h - mu)/s)   (checkpoint with sigreg.weight = 0)
    dense_sigreg y = G_theta((h - mu)/s)   (checkpoint with sigreg.weight > 0)

Probes must call the same ``encode_dense`` so that preprocessing is identical.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .data import torch_load
from .normalization import PCA_FORMAT


class RawFrontend(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.d_out = d_in

    def encode_dense(self, h: torch.Tensor) -> torch.Tensor:
        return h.float()


class PCAWhiteningFrontend(nn.Module):
    def __init__(self, pca: dict):
        super().__init__()
        if pca.get("format") != PCA_FORMAT:
            raise ValueError("not a PCA whitening file")
        self.register_buffer("mean", pca["mean"].float())
        self.register_buffer("scale", torch.tensor(float(pca["scale"])))
        inverse_std = (pca["eigenvalues"].float() + pca["epsilon"]).rsqrt()
        self.register_buffer("whitening", pca["eigenvectors"].float() * inverse_std)
        self.d_out = self.whitening.shape[1]

    def encode_dense(self, h: torch.Tensor) -> torch.Tensor:
        return ((h.float() - self.mean) / self.scale) @ self.whitening


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


def load_frontend(kind: str, path: str | Path | None = None, d_in: int = 0) -> nn.Module:
    if kind == "raw":
        frontend: nn.Module = RawFrontend(d_in)
    elif kind == "pca":
        frontend = PCAWhiteningFrontend(torch_load(path))
    elif kind in {"dense_ae", "dense_sigreg"}:
        frontend = DenseCheckpointFrontend(path)
    else:
        raise ValueError(f"unknown front-end {kind!r}")
    frontend.eval()
    for parameter in frontend.parameters():
        parameter.requires_grad_(False)
    return frontend
