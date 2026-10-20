"""Heatmap overlays and report figures."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def heatmap_overlay(image: np.ndarray, amap: np.ndarray, vmax: float,
                    alpha: float = 0.5) -> np.ndarray:
    """image: RGB float [0,1] (H,W,3); amap: (H,W). Returns RGB uint8."""
    norm = np.clip(amap / max(vmax, 1e-12), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    base = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)


def save_sample_grid(path: Path, images, recons, maps, masks, scores, labels, defect_types,
                     threshold: float, vmax: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.6 * n))
    axes = np.atleast_2d(axes)
    titles = ["Input", "Reconstruction", "Anomaly map", "Ground truth"]
    for i in range(n):
        pred = "ANOMALY" if scores[i] > threshold else "OK"
        ok = (scores[i] > threshold) == bool(labels[i])
        panels = [images[i], recons[i], heatmap_overlay(images[i], maps[i], vmax), masks[i]]
        for j, (ax, p) in enumerate(zip(axes[i], panels, strict=False)):
            ax.imshow(p, cmap="gray" if j == 3 else None, vmin=0, vmax=1 if j == 3 else None)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize=11)
        axes[i, 0].set_ylabel(defect_types[i], fontsize=10)
        axes[i, 2].set_xlabel(f"score {scores[i]:.4f} → {pred}", fontsize=9,
                              color="#1a7f37" if ok else "#cf222e")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_score_plots(path: Path, labels: np.ndarray, scores: np.ndarray, defect_types,
                     threshold: float, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(scores.min(), scores.max(), 40)
    ax1.hist(scores[labels == 0], bins=bins, alpha=0.7, label="normal", color="#2da44e")
    ax1.hist(scores[labels == 1], bins=bins, alpha=0.7, label="anomalous", color="#cf222e")
    ax1.axvline(threshold, color="black", ls="--", lw=1.2, label="calibrated threshold")
    ax1.set_xlabel("anomaly score")
    ax1.set_ylabel("images")
    ax1.set_title("Score distribution")
    ax1.legend()

    good = labels == 0
    for d in sorted(set(defect_types) - {"good"}):
        sel = good | np.array([t == d for t in defect_types])
        fpr, tpr, _ = roc_curve(labels[sel], scores[sel])
        ax2.plot(fpr, tpr, lw=1.6, label=d)
    fpr, tpr, _ = roc_curve(labels, scores)
    ax2.plot(fpr, tpr, color="black", lw=2.2, label="all defects")
    ax2.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
    ax2.set_xlabel("false positive rate")
    ax2.set_ylabel("true positive rate")
    ax2.set_title("ROC per defect type")
    ax2.legend(loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
