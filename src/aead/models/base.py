"""Shared convolutional encoder/decoder and the autoencoder interface."""

from __future__ import annotations

import torch
from torch import nn

from aead.scoring import reconstruction_loss


def _down(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True),
    )


def _up(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class ConvEncoder(nn.Module):
    """Strided-conv encoder: (B, 3, S, S) -> (B, C_last, S/2^n, S/2^n)."""

    def __init__(self, channels: list[int], in_channels: int = 3):
        super().__init__()
        chans = [in_channels, *channels]
        self.net = nn.Sequential(*[_down(a, b) for a, b in
                                   zip(chans[:-1], chans[1:], strict=True)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    """Mirror of ConvEncoder ending in a sigmoid, so outputs live in [0, 1]."""

    def __init__(self, channels: list[int], out_channels: int = 3):
        super().__init__()
        rev = list(reversed(channels))
        chans = [*rev, rev[-1]]
        self.net = nn.Sequential(*[_up(a, b) for a, b in
                                   zip(chans[:-1], chans[1:], strict=True)])
        self.head = nn.Sequential(nn.Conv2d(rev[-1], out_channels, 3, padding=1), nn.Sigmoid())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(z))


class BaseAutoencoder(nn.Module):
    """Conv encoder -> dense latent bottleneck -> conv decoder.

    The dense bottleneck forces a compact global representation, which is what makes
    the model fail to reconstruct unseen defects (the signal we score on).
    """

    def __init__(self, image_size: int, channels: list[int], latent_dim: int):
        super().__init__()
        self.encoder = ConvEncoder(channels)
        self.decoder = ConvDecoder(channels)
        side = image_size // 2 ** len(channels)
        self.feat_shape = (channels[-1], side, side)
        self.feat_dim = channels[-1] * side * side
        self.latent_dim = latent_dim

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)

    def decode_latent(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h.view(-1, *self.feat_shape))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction used for scoring."""
        return self.forward(x)["recon"]

    def loss(self, x: torch.Tensor, out: dict[str, torch.Tensor], ssim_weight: float
             ) -> dict[str, torch.Tensor]:
        rec = reconstruction_loss(x, out["recon"], ssim_weight)
        return {"total": rec, "recon": rec}
