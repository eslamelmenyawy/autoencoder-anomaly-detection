"""Unified predictor for PyTorch checkpoints (.pt), ONNX (.onnx) and OpenVINO IR (.xml).

Importing this module does not import PyTorch, so ONNX/OpenVINO deployments stay light.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aead.data import preprocess


@dataclass
class Prediction:
    score: float
    threshold: float
    is_anomaly: bool
    anomaly_map: np.ndarray = field(repr=False)  # (H, W) at model resolution
    latency_ms: float

    def to_dict(self) -> dict:
        return {"score": round(self.score, 6), "threshold": round(self.threshold, 6),
                "is_anomaly": self.is_anomaly, "label": "ANOMALY" if self.is_anomaly else "OK",
                "latency_ms": round(self.latency_ms, 2)}


class Predictor:
    """Load once, then call ``predict(image)`` with an RGB uint8/uint16/float array."""

    def __init__(self, run_fn, image_size: int, calibration: dict, backend: str,
                 model_name: str):
        self._run = run_fn
        self.image_size = image_size
        self.calibration = calibration
        self.threshold = float(calibration["threshold"])
        self.backend = backend
        self.model_name = model_name

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> Predictor:
        path = Path(path)
        if path.suffix == ".pt":
            return cls._load_torch(path, device)
        meta = json.loads((path.parent / "model.json").read_text())
        if path.suffix == ".onnx":
            import onnxruntime as ort

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
                if device.startswith("cuda") else ["CPUExecutionProvider"]
            sess = ort.InferenceSession(str(path), providers=providers)

            def run(x):
                amap, score, _ = sess.run(None, {"image": x})
                return amap, score
            backend = "onnxruntime"
        elif path.suffix == ".xml":
            import openvino as ov

            compiled = ov.Core().compile_model(str(path), device.upper()
                                               if device != "cpu" else "CPU")

            def run(x):
                res = compiled(x)
                return res[compiled.output(0)], res[compiled.output(1)]
            backend = "openvino"
        else:
            raise ValueError(f"Unsupported model file: {path} (use .pt, .onnx or .xml)")
        return cls(run, meta["image_size"], meta["calibration"], backend, meta["model"])

    @classmethod
    def _load_torch(cls, path: Path, device: str) -> Predictor:
        import torch

        from aead.engine import load_checkpoint, resolve_device

        dev = resolve_device(device if device != "cpu" else "cpu")
        detector, cfg, calib = load_checkpoint(path, dev)

        @torch.no_grad()
        def run(x):
            amap, score, _ = detector(torch.from_numpy(x).to(dev))
            return amap.cpu().numpy(), score.cpu().numpy()
        return cls(run, cfg.data.image_size, calib, f"pytorch-{dev.type}", cfg.model.name)

    def predict(self, image: np.ndarray) -> Prediction:
        x = preprocess(image, self.image_size).transpose(2, 0, 1)[None].astype(np.float32)
        t0 = time.perf_counter()
        amap, score = self._run(np.ascontiguousarray(x))
        latency = (time.perf_counter() - t0) * 1000
        s = float(np.asarray(score).reshape(-1)[0])
        return Prediction(s, self.threshold, s > self.threshold,
                          np.asarray(amap)[0, 0], latency)

    def predict_file(self, path: str | Path) -> Prediction:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB if img.shape[2] == 4
                               else cv2.COLOR_BGR2RGB)
        return self.predict(img)
