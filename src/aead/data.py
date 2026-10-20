"""Datasets in the MVTec AD folder layout.

    root/
    ├── train/good/*.png
    ├── test/good/*.png
    ├── test/<defect_type>/*.png
    └── ground_truth/<defect_type>/<image_stem>_mask.png   (optional)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int  # 0 = normal, 1 = anomalous
    defect_type: str
    mask_path: Path | None = None


def load_image(path: str | Path, size: int) -> np.ndarray:
    """Read an image as RGB float32 in [0, 1], resized to (size, size). Handles 8/16-bit."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return preprocess(img, size, bgr=True)


def preprocess(img: np.ndarray, size: int, bgr: bool = False) -> np.ndarray:
    """Convert any uint8/uint16/float image to RGB float32 [0, 1] at (size, size)."""
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    elif img.shape[2] == 4:
        img = img[..., :3]
    if bgr:
        img = img[..., ::-1]
    img = cv2.resize(np.ascontiguousarray(img), (size, size), interpolation=cv2.INTER_AREA)
    return np.clip(img, 0.0, 1.0)


def load_mask(path: Path | None, size: int) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros((size, size), dtype=np.float32)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.float32)


def _images_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def scan_dataset(root: str | Path, split: str) -> list[Sample]:
    """List samples for split 'train' (normal only) or 'test' (normal + defects)."""
    root = Path(root)
    if split == "train":
        samples = [Sample(p, 0, "good") for p in _images_in(root / "train" / "good")]
    elif split == "test":
        samples = []
        test_dir = root / "test"
        if test_dir.is_dir():
            for defect_dir in sorted(d for d in test_dir.iterdir() if d.is_dir()):
                defect = defect_dir.name
                for p in _images_in(defect_dir):
                    mask = None
                    if defect != "good":
                        mask = root / "ground_truth" / defect / f"{p.stem}_mask.png"
                    samples.append(Sample(p, int(defect != "good"), defect, mask))
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if not samples:
        raise FileNotFoundError(f"No images found for split '{split}' under {root}")
    return samples
