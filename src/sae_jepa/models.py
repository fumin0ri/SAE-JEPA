"""Dense reconstruction + SIGReg autoencoder (``model.type = dense_sigreg_ae``).

    x     = (h - mu) / s                 train-only mean vector, scalar scale
    y     = G_theta(x)                   Linear -> GELU -> Linear, no output activation
    x_hat = D_phi(y)                     one unconstrained Linear layer
    h_hat = s * x_hat + mu

There is no normalization layer, no sparsity mechanism, and no decoder column
norm constraint: the decoder scale is free so that a Gaussian-amplitude ``y``
can still reproduce the residual.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


ARCHITECTURE_ID = "dense_sigreg_ae_v1"

_ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


class DenseSIGRegAE(nn.Module):
    def __init__(self, cfg: ModelConfig, mean: torch.Tensor, scale: float):
        super().__init__()
        if cfg.d_in < 1:
            raise ValueError("model.d_in must be resolved before building the model")
        if mean.shape != (cfg.d_in,):
            raise ValueError("normalization mean does not match d_in")
        if not scale > 0:
            raise ValueError("normalization scale must be positive")
        if cfg.activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {cfg.activation!r}")
        self.cfg = cfg
        self.register_buffer("input_mean", mean.detach().float().clone())
        self.register_buffer("input_scale", torch.tensor(float(scale)))
        self.encoder = nn.Sequential(
            nn.Linear(cfg.d_in, cfg.d_hidden),
            _ACTIVATIONS[cfg.activation](),
            nn.Linear(cfg.d_hidden, cfg.d_latent),
        )
        self.decoder = nn.Linear(cfg.d_latent, cfg.d_in)

    def normalize(self, h: torch.Tensor) -> torch.Tensor:
        return (h.float() - self.input_mean) / self.input_scale

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() * self.input_scale + self.input_mean

    def encode_normalized(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode_dense(self, h: torch.Tensor) -> torch.Tensor:
        """Common stage-2 interface: residual ``h`` -> dense representation ``y``."""
        return self.encode_normalized(self.normalize(h))

    def decode_normalized(self, y: torch.Tensor) -> torch.Tensor:
        return self.decoder(y)

    def reconstruct(self, h: torch.Tensor) -> torch.Tensor:
        return self.denormalize(self.decode_normalized(self.encode_dense(h)))

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        if h.ndim != 2 or h.shape[-1] != self.cfg.d_in:
            raise ValueError("h must have shape [batch, d_in]")
        x = self.normalize(h)
        y = self.encode_normalized(x)
        return {"x": x, "y": y, "x_hat": self.decode_normalized(y)}


def build_model(cfg: ModelConfig, mean: torch.Tensor, scale: float) -> nn.Module:
    if cfg.type == "dense_sigreg_ae":
        return DenseSIGRegAE(cfg, mean, scale)
    raise ValueError(f"unknown model.type {cfg.type!r}")
