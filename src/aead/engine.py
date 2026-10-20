"""Training loop, threshold calibration and checkpoint I/O."""

from __future__ import annotations

import csv
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aead.config import Config
from aead.datasets import build_train_loaders
from aead.models import build_model
from aead.scoring import AnomalyDetector, ScoringHead

log = logging.getLogger("aead")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_detector(cfg: Config) -> AnomalyDetector:
    s = cfg.scoring
    head = ScoringHead(cfg.data.image_size, s.ssim_weight, s.blur_sigma, s.topk_ratio)
    return AnomalyDetector(build_model(cfg), head)


@torch.no_grad()
def collect_scores(detector: AnomalyDetector, loader: DataLoader, device: torch.device
                   ) -> tuple[np.ndarray, np.ndarray]:
    detector.eval()
    scores, maps = [], []
    for batch in loader:
        amap, score, _ = detector(batch["image"].to(device))
        scores.append(score.cpu().numpy())
        maps.append(amap.cpu().numpy())
    return np.concatenate(scores), np.concatenate(maps)


def calibrate(detector: AnomalyDetector, val_loader: DataLoader, device: torch.device,
              percentile: float) -> dict[str, float]:
    """Pick the decision threshold from held-out *normal* images only (no test leakage)."""
    scores, maps = collect_scores(detector, val_loader, device)
    return {
        "threshold": float(np.percentile(scores, percentile)),
        "threshold_percentile": percentile,
        "val_score_mean": float(scores.mean()),
        "val_score_std": float(scores.std()),
        "map_vis_max": float(np.percentile(maps, 99.9)) * 3.0,  # colour scale for heatmaps
    }


def _ssim_weight_at(cfg: Config, epoch: int) -> float:
    """Linear SSIM warm-up. Starting with pure MSE lets the model escape the trivial
    'predict the mean image' solution that SSIM alone has weak gradients against."""
    warm = cfg.train.ssim_warmup_epochs
    if warm <= 0:
        return cfg.train.ssim_weight
    return cfg.train.ssim_weight * min(1.0, max(0, epoch - 1) / warm)


def _run_epoch(model, loader, device, cfg: Config, ssim_weight: float, optimizer=None,
               scaler=None) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    n = 0
    use_amp = scaler is not None
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train), \
                torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)
            losses = model.loss(x, {k: v.float() for k, v in out.items()}, ssim_weight)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(losses["total"]).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                optimizer.step()
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item() * len(x)
        n += len(x)
    return {k: v / n for k, v in totals.items()}


def train(cfg: Config, run_dir: str | Path | None = None) -> Path:
    """Train, keep the best checkpoint by validation loss, calibrate it and save best.pt."""
    seed_everything(cfg.seed)
    device = resolve_device(cfg.train.device)
    run_dir = Path(run_dir or Path(cfg.output_dir) / cfg.experiment)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    train_loader, val_loader = build_train_loaders(cfg)
    detector = build_detector(cfg).to(device)
    model = detector.autoencoder
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model %s | %.2fM params | device %s | train %d / val %d images",
             cfg.model.name, n_params / 1e6, device, len(train_loader.dataset),
             len(val_loader.dataset))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    scaler = torch.amp.GradScaler("cuda") if cfg.train.amp and device.type == "cuda" else None

    best, best_epoch, history = float("inf"), -1, []
    t_start = time.time()
    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        w = _ssim_weight_at(cfg, epoch)
        tr = _run_epoch(model, train_loader, device, cfg, w, optimizer, scaler)
        # Validate with the final loss definition so checkpoints are comparable across epochs.
        va = _run_epoch(model, val_loader, device, cfg, cfg.train.ssim_weight)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr["total"], "val_loss": va["total"],
                        "lr": optimizer.param_groups[0]["lr"]})
        improved = va["total"] < best
        if improved:
            best, best_epoch = va["total"], epoch
            torch.save(model.state_dict(), run_dir / "_best_weights.pt")
        log.info("epoch %3d/%d | train %.5f | val %.5f %s| %.1fs", epoch, cfg.train.epochs,
                 tr["total"], va["total"], "* " if improved else "", time.time() - t0)
        if (epoch > cfg.train.ssim_warmup_epochs
                and epoch - best_epoch >= cfg.train.early_stopping_patience):
            log.info("Early stopping: no improvement for %d epochs",
                     cfg.train.early_stopping_patience)
            break

    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    model.load_state_dict(torch.load(run_dir / "_best_weights.pt", map_location=device))
    (run_dir / "_best_weights.pt").unlink()
    calibration = calibrate(detector, val_loader, device, cfg.scoring.threshold_percentile)
    save_checkpoint(run_dir / "best.pt", detector, cfg, calibration,
                    {"best_epoch": best_epoch, "best_val_loss": best,
                     "train_time_s": round(time.time() - t_start, 1),
                     "params": n_params})
    log.info("Saved %s (best epoch %d, threshold %.5f)", run_dir / "best.pt", best_epoch,
             calibration["threshold"])
    return run_dir


def save_checkpoint(path: Path, detector: AnomalyDetector, cfg: Config,
                    calibration: dict, info: dict) -> None:
    torch.save({"config": cfg.to_dict(), "state_dict": detector.autoencoder.state_dict(),
                "calibration": calibration, "info": info}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"
                    ) -> tuple[AnomalyDetector, Config, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    detector = build_detector(cfg)
    detector.autoencoder.load_state_dict(ckpt["state_dict"])
    detector.to(device).eval()
    return detector, cfg, ckpt["calibration"]
