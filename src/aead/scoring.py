"""Anomaly maps and image scores from reconstructions.

Everything here is a plain ``nn.Module`` built from convolutions, so the full
detector (autoencoder + scoring) exports to a single ONNX graph.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def gaussian_kernel1d(sigma: float, radius: int | None = None) -> torch.Tensor:
    radius = radius if radius is not None else max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def separable_kernels(k1d: torch.Tensor, channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise horizontal/vertical kernels with concrete shapes (ONNX-friendly)."""
    n = k1d.numel()
    return (k1d.view(1, 1, 1, n).repeat(channels, 1, 1, 1).contiguous(),
            k1d.view(1, 1, n, 1).repeat(channels, 1, 1, 1).contiguous())


def _separable_blur(x: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    """Depthwise Gaussian blur with reflect padding. x: (B, C, H, W)."""
    c, r = kx.shape[0], kx.shape[-1] // 2
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kx, groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), ky, groups=c)


SSIM_SIGMA, SSIM_RADIUS = 1.5, 5


def ssim_map(x: torch.Tensor, y: torch.Tensor, kernels=None,
             c1: float = 0.01**2, c2: float = 0.03**2) -> torch.Tensor:
    """Per-pixel, per-channel SSIM for images in [0, 1]. Returns (B, C, H, W)."""
    if kernels is None:
        kernels = separable_kernels(gaussian_kernel1d(SSIM_SIGMA, SSIM_RADIUS).to(x),
                                    x.shape[1])
    kx, ky = kernels
    mu_x, mu_y = _separable_blur(x, kx, ky), _separable_blur(y, kx, ky)
    sxx = _separable_blur(x * x, kx, ky) - mu_x**2
    syy = _separable_blur(y * y, kx, ky) - mu_y**2
    sxy = _separable_blur(x * y, kx, ky) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2)
    return num / den


def reconstruction_loss(x: torch.Tensor, recon: torch.Tensor, ssim_weight: float
                        ) -> torch.Tensor:
    mse = F.mse_loss(recon, x)
    if ssim_weight <= 0:
        return mse
    return (1 - ssim_weight) * mse + ssim_weight * (1 - ssim_map(x, recon).mean())


class ScoringHead(nn.Module):
    """Turns (input, reconstruction) into a smoothed anomaly map and an image score."""

    def __init__(self, image_size: int, ssim_weight: float = 0.5, blur_sigma: float = 4.0,
                 topk_ratio: float = 0.01):
        super().__init__()
        self.ssim_weight = float(ssim_weight)
        self.topk = max(1, int(round(topk_ratio * image_size * image_size)))
        self.use_blur = blur_sigma > 0
        sx, sy = separable_kernels(gaussian_kernel1d(SSIM_SIGMA, SSIM_RADIUS), 3)
        self.register_buffer("ssim_kx", sx)
        self.register_buffer("ssim_ky", sy)
        bx, by = separable_kernels(gaussian_kernel1d(blur_sigma if self.use_blur else 1.0), 1)
        self.register_buffer("blur_kx", bx)
        self.register_buffer("blur_ky", by)

    def forward(self, x: torch.Tensor, recon: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        l2 = ((x - recon) ** 2).mean(dim=1, keepdim=True)
        amap = (1 - self.ssim_weight) * l2
        if self.ssim_weight > 0:
            ssim = ssim_map(x, recon, (self.ssim_kx, self.ssim_ky))
            amap = amap + self.ssim_weight * ((1 - ssim) / 2).mean(dim=1, keepdim=True)
        if self.use_blur:
            amap = _separable_blur(amap, self.blur_kx, self.blur_ky)
        score = torch.topk(amap.flatten(1), self.topk, dim=1).values.mean(dim=1)
        return amap, score


class AnomalyDetector(nn.Module):
    """Autoencoder + scoring head: image batch -> (anomaly_map, score, reconstruction)."""

    def __init__(self, autoencoder: nn.Module, head: ScoringHead):
        super().__init__()
        self.autoencoder = autoencoder
        self.head = head

    def forward(self, x: torch.Tensor):
        recon = self.autoencoder.reconstruct(x)
        amap, score = self.head(x, recon)
        return amap, score, recon
