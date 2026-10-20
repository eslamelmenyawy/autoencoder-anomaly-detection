"""Deterministic convolutional autoencoder."""

from __future__ import annotations

import torch
from torch import nn

from aead.models.base import BaseAutoencoder


class ConvAutoencoder(BaseAutoencoder):
    def __init__(self, image_size: int, channels: list[int], latent_dim: int):
        super().__init__(image_size, channels, latent_dim)
        self.to_latent = nn.Linear(self.feat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.feat_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.to_latent(self.encode_features(x))
        return {"recon": self.decode_latent(self.from_latent(z)), "z": z}
