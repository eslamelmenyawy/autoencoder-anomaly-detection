"""Variational autoencoder (Kingma & Welling, 2014) with a spatial latent.

The encoder predicts a Gaussian posterior (mu, log-variance) at every position of the
8x8 latent grid, with only a few channels. The narrow channel count is the bottleneck;
the KL term keeps the latent compact and smooth, so unseen defects reconstruct poorly.
"""

from __future__ import annotations

import torch
from torch import nn

from aead.models.base import BaseAutoencoder
from aead.scoring import reconstruction_loss


class VariationalAutoencoder(BaseAutoencoder):
    def __init__(self, image_size: int, channels: list[int], latent_dim: int,
                 kl_weight: float = 0.01, latent_channels: int = 16):
        super().__init__(image_size, channels, latent_dim)
        self.kl_weight = kl_weight
        c = channels[-1]
        self.to_mu = nn.Conv2d(c, latent_channels, 1)
        self.to_logvar = nn.Conv2d(c, latent_channels, 1)
        # Start near-deterministic (sigma ~ 0.05): large initial sampling noise makes the
        # decoder ignore z and the model collapses to the mean image.
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, -6.0)
        self.from_latent = nn.Conv2d(latent_channels, c, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = self.to_mu(h), self.to_logvar(h).clamp(-10, 10)
        # Sample during training; use the posterior mean at inference for stable scores.
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return {"recon": self.decoder(self.from_latent(z)), "mu": mu, "logvar": logvar}

    def loss(self, x, out, ssim_weight):
        rec = reconstruction_loss(x, out["recon"], ssim_weight)
        mu, logvar = out["mu"], out["logvar"]
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2, 3)).mean()
        kl = kl / x[0].numel()  # per-pixel scale, comparable to the reconstruction term
        return {"total": rec + self.kl_weight * kl, "recon": rec, "kl": kl}
