"""Variational autoencoder (Kingma & Welling, 2014)."""

from __future__ import annotations

import torch
from torch import nn

from aead.models.base import BaseAutoencoder
from aead.scoring import reconstruction_loss


class VariationalAutoencoder(BaseAutoencoder):
    def __init__(self, image_size: int, channels: list[int], latent_dim: int,
                 kl_weight: float = 1.0):
        super().__init__(image_size, channels, latent_dim)
        self.kl_weight = kl_weight
        self.fc_mu = nn.Linear(self.feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.feat_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode_features(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h).clamp(-10, 10)
        # Sample during training; use the posterior mean at inference for stable scores.
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return {"recon": self.decode_latent(self.from_latent(z)), "mu": mu, "logvar": logvar}

    def loss(self, x, out, ssim_weight):
        rec = reconstruction_loss(x, out["recon"], ssim_weight)
        mu, logvar = out["mu"], out["logvar"]
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        kl = kl / x[0].numel()  # per-pixel scale, comparable to the reconstruction term
        return {"total": rec + self.kl_weight * kl, "recon": rec, "kl": kl}
