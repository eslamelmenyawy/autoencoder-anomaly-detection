"""Image- and pixel-level anomaly detection metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def best_f1(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """F1-optimal threshold. Oracle metric (uses test labels) — report alongside, not instead."""
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.argmax(f1[:-1])) if len(t) else 0
    return float(f1[i]), float(t[i]) if len(t) else float("nan")


def classification_at(y: np.ndarray, s: np.ndarray, threshold: float) -> dict[str, float]:
    pred = s > threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(len(y), 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "false_positive_rate": fp / max(fp + tn, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def compute_metrics(labels: np.ndarray, scores: np.ndarray, defect_types: list[str],
                    threshold: float, maps: np.ndarray | None = None,
                    masks: np.ndarray | None = None, max_pixels: int = 4_000_000,
                    seed: int = 0) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    f1_best, t_best = best_f1(labels, scores)
    out: dict = {
        "image_auroc": _safe_auc(labels, scores),
        "image_ap": float(average_precision_score(labels, scores)) if labels.any() else None,
        "calibrated": classification_at(labels, scores, threshold),
        "oracle_best_f1": {"f1": f1_best, "threshold": t_best},
        "n_images": int(len(labels)),
        "n_anomalous": int(labels.sum()),
    }
    good = np.array([d == "good" for d in defect_types])
    out["per_defect_auroc"] = {}
    for d in sorted(set(defect_types) - {"good"}):
        sel = good | np.array([t == d for t in defect_types])
        out["per_defect_auroc"][d] = _safe_auc(labels[sel], scores[sel])

    if maps is not None and masks is not None and masks.any():
        m, g = maps.reshape(-1), (masks.reshape(-1) > 0.5).astype(np.uint8)
        if m.size > max_pixels:  # subsample to bound memory on large test sets
            idx = np.random.default_rng(seed).choice(m.size, max_pixels, replace=False)
            m, g = m[idx], g[idx]
        out["pixel_auroc"] = _safe_auc(g, m)
    return out
