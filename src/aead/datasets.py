"""PyTorch datasets and loaders (training/evaluation only)."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from aead.config import Config
from aead.data import Sample, load_image, load_mask, scan_dataset


class InspectionDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int, augment: bool = False):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image = torch.from_numpy(load_image(s.path, self.image_size).transpose(2, 0, 1).copy())
        mask = torch.from_numpy(load_mask(s.mask_path, self.image_size))[None]
        if self.augment:
            if torch.rand(1).item() < 0.5:
                image, mask = image.flip(-1), mask.flip(-1)
            if torch.rand(1).item() < 0.5:
                image, mask = image.flip(-2), mask.flip(-2)
        return {"image": image, "mask": mask, "label": s.label,
                "defect_type": s.defect_type, "path": str(s.path)}


def split_train_val(samples: list[Sample], val_split: float, seed: int
                    ) -> tuple[list[Sample], list[Sample]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n_val = max(1, int(round(len(samples) * val_split)))
    val = [samples[i] for i in idx[:n_val]]
    train = [samples[i] for i in idx[n_val:]]
    return train, val


def build_train_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    train_s, val_s = split_train_val(scan_dataset(cfg.data.root, "train"),
                                     cfg.data.val_split, cfg.seed)
    size, bs, nw = cfg.data.image_size, cfg.train.batch_size, cfg.data.num_workers
    train_ds = InspectionDataset(train_s, size, augment=cfg.data.augment)
    val_ds = InspectionDataset(val_s, size)
    return (DataLoader(train_ds, bs, shuffle=True, num_workers=nw, drop_last=len(train_ds) > bs),
            DataLoader(val_ds, bs, shuffle=False, num_workers=nw))


def build_test_loader(root: str, image_size: int, batch_size: int = 16,
                      num_workers: int = 0) -> DataLoader:
    ds = InspectionDataset(scan_dataset(root, "test"), image_size)
    return DataLoader(ds, batch_size, shuffle=False, num_workers=num_workers)
