import numpy as np
import torch

from aead.metrics import classification_at, compute_metrics
from aead.scoring import ScoringHead, ssim_map


def test_ssim_of_identical_images_is_one():
    x = torch.rand(2, 3, 32, 32)
    assert torch.allclose(ssim_map(x, x), torch.ones_like(x), atol=1e-4)


def test_perfect_reconstruction_scores_zero():
    x = torch.rand(1, 3, 64, 64)
    amap, score = ScoringHead(64)(x, x)
    assert amap.abs().max() < 1e-4 and score.item() < 1e-4


def test_anomaly_map_peaks_at_the_defect():
    x = torch.full((1, 3, 64, 64), 0.5)
    corrupted = x.clone()
    corrupted[..., 40:48, 10:18] = 0.0  # defect the "model" failed to reconstruct
    amap, score = ScoringHead(64, ssim_weight=0.5, blur_sigma=2)(corrupted, x)
    y, xx = np.unravel_index(amap[0, 0].argmax().item(), (64, 64))
    assert 38 <= y <= 50 and 8 <= xx <= 20
    assert score.item() > 0.05


def test_metrics_on_perfect_separation():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.7])
    types = ["good"] * 3 + ["scratch", "crack", "crack"]
    m = compute_metrics(labels, scores, types, threshold=0.5)
    assert m["image_auroc"] == 1.0
    assert m["per_defect_auroc"] == {"crack": 1.0, "scratch": 1.0}
    assert m["calibrated"]["f1"] == 1.0


def test_classification_counts():
    c = classification_at(np.array([0, 1, 1, 0]), np.array([0.6, 0.7, 0.2, 0.1]), 0.5)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 1)
