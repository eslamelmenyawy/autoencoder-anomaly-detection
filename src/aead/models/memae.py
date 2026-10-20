"""Memory-augmented autoencoder (Gong et al., ICCV 2019).

Each spatial position of the encoder's feature map is rebuilt from a sparse mix of
learned memory items that represent *normal* patterns. Anomalies cannot be expressed by
the memory, so they reconstruct poorly — which sharpens the anomaly signal versus a
plain autoencoder that can "generalise too well" and reconstruct defects.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from aead.models.base import BaseAutoencoder
from aead.scoring import reconstruction_loss


class MemoryModule(nn.Module):
    """Content-addressable memory with hard-shrinkage sparse addressing."""

    def __init__(self, memory_size: int, dim: int, shrink_threshold: float = 0.0025,
                 temperature: float = 0.1):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, dim) / dim**0.5)
        self.shrink_threshold = shrink_threshold
        self.temperature = temperature

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z: (N, D) queries -> (N, D) reconstructions from memory, (N, M) weights."""
        # Cosine-similarity addressing, as in the paper: bounded logits keep the softmax
        # from collapsing onto a single slot at initialisation.
        sim = F.normalize(z, dim=1) @ F.normalize(self.memory, dim=1).t()
        w = F.softmax(sim / self.temperature, dim=1)
        if self.shrink_threshold > 0:  # hard shrinkage -> sparse addressing
            shifted = w - self.shrink_threshold
            w = F.relu(shifted) * w / (shifted.abs() + 1e-12)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        return w @ self.memory, w


class MemAutoencoder(BaseAutoencoder):
    """Conv encoder -> per-position memory addressing -> conv decoder."""

    def __init__(self, image_size: int, channels: list[int], latent_dim: int,
                 memory_size: int = 100, shrink_threshold: float = 0.0025,
                 entropy_weight: float = 0.0002):
        super().__init__(image_size, channels, latent_dim)
        self.entropy_weight = entropy_weight
        c = channels[-1]
        self.to_latent = nn.Conv2d(c, latent_dim, 1)
        self.memory = MemoryModule(memory_size, latent_dim, shrink_threshold)
        self.from_latent = nn.Conv2d(latent_dim, c, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.to_latent(self.encoder(x))  # (B, D, h, w)
        b, d, h, w = z.shape
        queries = z.permute(0, 2, 3, 1).reshape(b * h * w, d)
        z_hat, attn = self.memory(queries)
        z_hat = z_hat.reshape(b, h, w, d).permute(0, 3, 1, 2)
        return {"recon": self.decoder(self.from_latent(z_hat)), "attention": attn}

    def loss(self, x, out, ssim_weight):
        rec = reconstruction_loss(x, out["recon"], ssim_weight)
        w = out["attention"]
        entropy = (-w * torch.log(w + 1e-12)).sum(dim=1).mean()
        return {"total": rec + self.entropy_weight * entropy, "recon": rec,
                "entropy": entropy}
