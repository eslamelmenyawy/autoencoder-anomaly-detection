"""REST inference service.

    AEAD_MODEL=runs/conv_ae/export/model.onnx uvicorn aead.api:app --port 8000
"""

from __future__ import annotations

import base64
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from aead import __version__
from aead.inference import Predictor
from aead.visualize import heatmap_overlay


def create_app(model_path: str | None = None) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        path = model_path or os.environ.get("AEAD_MODEL")
        if not path:
            raise RuntimeError("Set AEAD_MODEL to a .onnx, .xml or .pt model file")
        state["predictor"] = Predictor.load(path, os.environ.get("AEAD_DEVICE", "cpu"))
        yield
        state.clear()

    app = FastAPI(title="Autoencoder Anomaly Detection API", version=__version__,
                  description="Upload an image, get an anomaly score, decision and heatmap.",
                  lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/model")
    def model_info() -> dict:
        p: Predictor = state["predictor"]
        return {"model": p.model_name, "backend": p.backend, "image_size": p.image_size,
                "threshold": p.threshold}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...),
                      heatmap: bool = Query(False, description="Return a PNG heatmap overlay")
                      ) -> dict:
        data = np.frombuffer(await file.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB if img.shape[2] == 4
                               else cv2.COLOR_BGR2RGB)
        p: Predictor = state["predictor"]
        pred = p.predict(img)
        result = {"filename": file.filename, **pred.to_dict()}
        if heatmap:
            from aead.data import preprocess

            small = preprocess(img, p.image_size)
            overlay = heatmap_overlay(small, pred.anomaly_map,
                                      p.calibration.get("map_vis_max", 1.0))
            ok, png = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            result["heatmap_png_base64"] = base64.b64encode(png.tobytes()).decode()
        return result

    return app


app = create_app() if os.environ.get("AEAD_MODEL") else None
