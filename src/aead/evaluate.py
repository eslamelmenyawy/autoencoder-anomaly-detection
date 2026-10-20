"""Evaluate a trained checkpoint on the test split and write a report."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from aead.datasets import build_test_loader
from aead.engine import load_checkpoint, resolve_device
from aead.metrics import compute_metrics
from aead.visualize import save_sample_grid, save_score_plots

log = logging.getLogger("aead")


@torch.no_grad()
def measure_latency(detector, image_size: int, device: torch.device, runs: int = 30
                    ) -> float:
    x = torch.rand(1, 3, image_size, image_size, device=device)
    for _ in range(5):
        detector(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        detector(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


@torch.no_grad()
def evaluate(checkpoint: str | Path, data_root: str | None = None,
             out_dir: str | Path | None = None, device: str = "auto",
             n_examples: int = 6) -> dict:
    dev = resolve_device(device)
    detector, cfg, calib = load_checkpoint(checkpoint, dev)
    root = data_root or cfg.data.root
    out_dir = Path(out_dir or Path(checkpoint).parent / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = build_test_loader(root, cfg.data.image_size, num_workers=cfg.data.num_workers)
    scores, labels, types, maps, masks, keep = [], [], [], [], [], []
    for batch in loader:
        amap, score, recon = detector(batch["image"].to(dev))
        scores.append(score.cpu().numpy())
        labels.append(batch["label"].numpy())
        types += list(batch["defect_type"])
        maps.append(amap[:, 0].cpu().numpy().astype(np.float32))
        masks.append(batch["mask"][:, 0].numpy().astype(np.uint8))
        keep.append((batch["image"].numpy(), recon.cpu().numpy()))

    scores, labels = np.concatenate(scores), np.concatenate(labels)
    maps, masks = np.concatenate(maps), np.concatenate(masks)
    metrics = compute_metrics(labels, scores, types, calib["threshold"], maps, masks)
    metrics["latency_ms_per_image"] = round(measure_latency(detector, cfg.data.image_size,
                                                            dev), 2)
    metrics["device"] = str(dev)
    metrics["model"] = cfg.model.name
    metrics["params_millions"] = round(sum(p.numel() for p in detector.parameters()) / 1e6, 2)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Figures: score distribution/ROC and a grid with one example per defect type.
    save_score_plots(out_dir / "scores_roc.png", labels, scores, types, calib["threshold"],
                     f"{cfg.model.name}  |  image AUROC {metrics['image_auroc']:.3f}")
    imgs = np.concatenate([k[0] for k in keep]).transpose(0, 2, 3, 1)
    recs = np.concatenate([k[1] for k in keep]).transpose(0, 2, 3, 1)
    order = sorted(set(types), key=lambda d: (d != "good", d))
    picks = [types.index(d) for d in order][:n_examples]
    save_sample_grid(out_dir / "examples.png", imgs[picks], recs[picks], maps[picks],
                     masks[picks], scores[picks], labels[picks], [types[i] for i in picks],
                     calib["threshold"], calib["map_vis_max"])

    log.info("image AUROC %.4f | pixel AUROC %s | F1@calibrated %.3f | %.2f ms/img (%s)",
             metrics["image_auroc"], f"{metrics.get('pixel_auroc', float('nan')):.4f}",
             metrics["calibrated"]["f1"], metrics["latency_ms_per_image"], dev)
    log.info("Report written to %s", out_dir)
    return metrics
