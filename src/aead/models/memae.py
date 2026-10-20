"""Memory-augmented autoencoder (Gong et al., ICCV 2019).

The latent code is replaced by a sparse combination of learned memory items that
represent *normal* patterns. Anomalies cannot be expressed by the memory, so they
reconstruct poorly — which sharpens the anomaly signal versus a plain autoencoder.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from aead.models.base import BaseAutoencoder
from aead.scoring import reconstruction_loss


class MemoryModule(nn.Module):
    def __init__(self, memory_size: int, dim: int, shrink_threshold: float = 0.0025):
        super().__init__()
        self.memory = nn.Parameter(torch.empty(memory_size, dim))
        nn.init.kaiming_uniform_(self.memory)
        self.shrink_threshold = shrink_threshold

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = F.softmax(z @ self.memory.t(), dim=1)
        if self.shrink_threshold > 0:  # hard shrinkage -> sparse addressing
            shifted = w - self.shrink_threshold
            w = F.relu(shifted) * w / (shifted.abs() + 1e-12)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        return w @ self.memory, w


class MemAutoencoder(BaseAutoencoder):
    def __init__(self, image_size: int, channels: list[int], latent_dim: int,
                 memory_size: int = 100, shrink_threshold: float = 0.0025,
                 entropy_weight: float = 0.0002):
        super().__init__(image_size, channels, latent_dim)
        self.entropy_weight = entropy_weight
        self.to_latent = nn.Linear(self.feat_dim, latent_dim)
        self.memory = MemoryModule(memory_size, latent_dim, shrink_threshold)
        self.from_latent = nn.Linear(latent_dim, self.feat_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.to_latent(self.encode_features(x))
        z_hat, w = self.memory(z)
        return {"recon": self.decode_latent(self.from_latent(z_hat)), "attention": w}

    def loss(self, x, out, ssim_weight):
        rec = reconstruction_loss(x, out["recon"], ssim_weight)
        w = out["attention"]
        entropy = (-w * torch.log(w + 1e-12)).sum(dim=1).mean()
        return {"total": rec + self.entropy_weight * entropy, "recon": rec,
                "entropy": entropy}
