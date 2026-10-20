"""Synthetic perforated-metal inspection dataset with pixel-accurate defect masks.

Lets you train, evaluate and demo the full pipeline in minutes without downloading
anything. Four defect types are generated: scratch, crack, dent and stain.
The output follows the MVTec AD layout, so real datasets are a drop-in replacement.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

DEFECTS = ("scratch", "crack", "dent", "stain")


def _surface(rng: np.random.Generator, size: int) -> np.ndarray:
    """Perforated brushed-metal plate (RGB float in [0, 1]).

    A regular hole lattice with random phase, fine brushed grain and a soft illumination
    gradient: structured enough for an autoencoder to learn, varied enough to be realistic.
    """
    grain = cv2.GaussianBlur(rng.normal(size=(size, size)).astype(np.float32), (0, 0),
                             sigmaX=8, sigmaY=0.7)
    grain = (grain - grain.mean()) / (grain.std() + 1e-6)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    angle = rng.uniform(0, 2 * np.pi)
    light = np.cos(angle) * xx + np.sin(angle) * yy
    gray = 0.62 + 0.012 * grain + 0.08 * (light - 0.5)

    holes = np.zeros((size, size), np.float32)
    pitch = size // 6
    ox, oy = rng.uniform(0, pitch, 2)
    radius = pitch * 0.22
    for cy in np.arange(oy - pitch, size + pitch, pitch):
        for cx in np.arange(ox - pitch, size + pitch, pitch):
            cv2.circle(holes, (int(cx), int(cy)), int(radius), 1.0, -1, cv2.LINE_AA)
    holes = cv2.GaussianBlur(holes, (0, 0), 1.2)
    gray = gray * (1 - 0.75 * holes)
    tint = np.array([0.98, 0.97, 0.94], dtype=np.float32)  # RGB
    return np.clip(gray[..., None] * tint, 0, 1)


def _blend(img: np.ndarray, mask: np.ndarray, color, strength: float, soften: float = 1.0):
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), soften) * strength
    color = np.asarray(color, dtype=np.float32)
    return img * (1 - alpha[..., None]) + color * alpha[..., None]


def _scratch(img, rng, size):
    mask = np.zeros((size, size), np.uint8)
    length = rng.uniform(0.25, 0.5) * size
    cx, cy = rng.uniform(0.25, 0.75, 2) * size
    a = rng.uniform(0, np.pi)
    p1 = (int(cx - np.cos(a) * length / 2), int(cy - np.sin(a) * length / 2))
    p2 = (int(cx + np.cos(a) * length / 2), int(cy + np.sin(a) * length / 2))
    cv2.line(mask, p1, p2, 1, int(rng.integers(2, 4)), cv2.LINE_AA)
    shade = rng.choice([0.85, 0.22])  # bright or dark scratch
    return _blend(img, mask, (shade,) * 3, 0.9, 0.7), mask


def _crack(img, rng, size):
    mask = np.zeros((size, size), np.uint8)
    pt = rng.uniform(0.3, 0.7, 2) * size
    heading = rng.uniform(0, 2 * np.pi)
    pts = [pt.copy()]
    for _ in range(int(rng.integers(8, 14))):
        heading += rng.normal(0, 0.6)
        pt = pt + 7 * np.array([np.cos(heading), np.sin(heading)])
        pts.append(pt.copy())
    cv2.polylines(mask, [np.array(pts, np.int32)], False, 1, int(rng.integers(2, 4)))
    return _blend(img, mask, (0.12, 0.11, 0.10), 0.95, 0.6), mask


def _dent(img, rng, size):
    mask = np.zeros((size, size), np.uint8)
    center = tuple(int(v) for v in rng.uniform(0.2, 0.8, 2) * size)
    axes = tuple(int(v) for v in rng.uniform(5, 12, 2))
    cv2.ellipse(mask, center, axes, rng.uniform(0, 180), 0, 360, 1, -1)
    return _blend(img, mask, (0.25, 0.25, 0.26), 0.8, 2.0), mask


def _stain(img, rng, size):
    mask = np.zeros((size, size), np.uint8)
    center = rng.uniform(0.2, 0.8, 2) * size
    for _ in range(int(rng.integers(3, 6))):  # irregular blob from overlapping circles
        c = center + rng.normal(0, 5, 2)
        cv2.circle(mask, (int(c[0]), int(c[1])), int(rng.integers(5, 11)), 1, -1)
    return _blend(img, mask, (0.62, 0.32, 0.12), 0.75, 2.0), mask  # rust-coloured


_DEFECT_FNS: dict[str, Callable] = {"scratch": _scratch, "crack": _crack, "dent": _dent,
                                    "stain": _stain}


def _save(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def generate_dataset(out: str | Path, n_train: int = 400, n_test_good: int = 60,
                     n_per_defect: int = 25, size: int = 256, seed: int = 0) -> Path:
    """Write a synthetic dataset in MVTec AD layout and return its root."""
    out = Path(out)
    rng = np.random.default_rng(seed)
    for i in range(n_train):
        _save(out / "train" / "good" / f"{i:03d}.png", _surface(rng, size))
    for i in range(n_test_good):
        _save(out / "test" / "good" / f"{i:03d}.png", _surface(rng, size))
    for defect in DEFECTS:
        for i in range(n_per_defect):
            img, mask = _DEFECT_FNS[defect](_surface(rng, size), rng, size)
            _save(out / "test" / defect / f"{i:03d}.png", np.clip(img, 0, 1))
            mpath = out / "ground_truth" / defect / f"{i:03d}_mask.png"
            mpath.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mpath), (mask > 0).astype(np.uint8) * 255)
    return out
