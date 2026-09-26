"""Per-token nonnegative Top-K SAE with unit-norm decoder columns."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TopKSAE(nn.Module):
    def __init__(self, d_in: int, dictionary_size: int, k: int):
        super().__init__()
        if d_in < 1 or not 1 <= k <= dictionary_size:
            raise ValueError("require d_in > 0 and 1 <= k <= dictionary_size")
        self.k = k
        self.encoder = nn.Linear(d_in, dictionary_size)
        self.decoder = nn.Linear(dictionary_size, d_in, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_in))
        with torch.no_grad():
            self.decoder.weight.copy_(F.normalize(torch.randn_like(self.decoder.weight), dim=0))
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()

    def forward(self, x):
        pre = self.encoder(x - self.bias)
        values, indices = torch.topk(pre, self.k, dim=-1)
        # ReLU after selection permits fewer than K positive features.
        values = values.relu()
        z = torch.zeros_like(pre).scatter(-1, indices, values)
        return self.decoder(z) + self.bias, z

    @torch.no_grad()
    def project_decoder_gradient(self):
        w, grad = self.decoder.weight, self.decoder.weight.grad
        if grad is not None:
            grad.sub_(w * (grad * w).sum(0, keepdim=True))

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))
