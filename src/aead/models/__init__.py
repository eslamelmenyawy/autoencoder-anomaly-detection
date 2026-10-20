"""Model registry."""

from __future__ import annotations

from aead.config import Config
from aead.models.base import BaseAutoencoder
from aead.models.conv_ae import ConvAutoencoder
from aead.models.memae import MemAutoencoder
from aead.models.vae import VariationalAutoencoder

__all__ = ["BaseAutoencoder", "ConvAutoencoder", "VariationalAutoencoder", "MemAutoencoder",
           "build_model"]


def build_model(cfg: Config) -> BaseAutoencoder:
    m, size = cfg.model, cfg.data.image_size
    if m.name == "conv_ae":
        return ConvAutoencoder(size, m.channels, m.latent_dim)
    if m.name == "vae":
        return VariationalAutoencoder(size, m.channels, m.latent_dim, m.kl_weight)
    if m.name == "memae":
        return MemAutoencoder(size, m.channels, m.latent_dim, m.memory_size,
                              m.shrink_threshold, m.entropy_weight)
    raise ValueError(f"Unknown model: {m.name}")
