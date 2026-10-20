"""Typed experiment configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data/synthetic"
    image_size: int = 128
    val_split: float = 0.1
    augment: bool = True
    num_workers: int = 0


@dataclass
class ModelConfig:
    name: str = "conv_ae"  # conv_ae | vae | memae
    channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    latent_dim: int = 128
    kl_weight: float = 0.01  # VAE only (beta; larger values cause posterior collapse)
    memory_size: int = 100  # MemAE only
    shrink_threshold: float = 0.0025  # MemAE only
    entropy_weight: float = 0.0002  # MemAE only


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    ssim_weight: float = 0.5  # weight of (1 - SSIM) in the reconstruction loss
    ssim_warmup_epochs: int = 10  # ramp SSIM in after pure-MSE epochs (avoids mean collapse)
    early_stopping_patience: int = 10
    device: str = "auto"  # auto | cpu | cuda | mps
    amp: bool = True  # mixed precision on CUDA


@dataclass
class ScoringConfig:
    ssim_weight: float = 0.5  # 0 = pure L2 residual, 1 = pure SSIM residual
    blur_sigma: float = 4.0
    topk_ratio: float = 0.01  # image score = mean of the top-k% anomaly-map pixels
    threshold_percentile: float = 99.0  # calibrated on held-out normal images


@dataclass
class Config:
    experiment: str = "conv_ae"
    seed: int = 42
    output_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        sections = {"data": DataConfig, "model": ModelConfig, "train": TrainConfig,
                    "scoring": ScoringConfig}
        kwargs: dict[str, Any] = {}
        top_level = {f.name for f in fields(cls)}
        for key, value in raw.items():
            if key not in top_level:
                raise KeyError(f"Unknown config key: {key!r}")
            if key in sections:
                known = {f.name for f in fields(sections[key])}
                unknown = set(value or {}) - known
                if unknown:
                    raise KeyError(f"Unknown keys in '{key}': {sorted(unknown)}")
                kwargs[key] = sections[key](**(value or {}))
            else:
                kwargs[key] = value
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        downsample = 2 ** len(self.model.channels)
        if self.data.image_size % downsample:
            raise ValueError(f"image_size ({self.data.image_size}) must be divisible by "
                             f"{downsample} for {len(self.model.channels)} encoder stages")
        if self.model.name not in {"conv_ae", "vae", "memae"}:
            raise ValueError(f"Unknown model '{self.model.name}'")
        if not 0.0 <= self.scoring.ssim_weight <= 1.0:
            raise ValueError("scoring.ssim_weight must be in [0, 1]")
