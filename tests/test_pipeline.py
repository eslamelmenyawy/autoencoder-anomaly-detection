"""End-to-end: synthetic data -> train -> evaluate -> export -> predict -> REST API."""

import subprocess
import sys

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from aead.api import create_app
from aead.data import scan_dataset
from aead.engine import train
from aead.evaluate import evaluate
from aead.export import export
from aead.inference import Predictor


def test_dataset_layout(tiny_dataset):
    train_s, test_s = scan_dataset(tiny_dataset, "train"), scan_dataset(tiny_dataset, "test")
    assert len(train_s) == 24
    assert {s.defect_type for s in test_s} == {"good", "scratch", "crack", "dent", "stain"}
    defects = [s for s in test_s if s.label == 1]
    assert all(s.mask_path is not None and s.mask_path.exists() for s in defects)
    mask = cv2.imread(str(defects[0].mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask.max() == 255  # every defect has a non-empty mask


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory, tiny_dataset):
    from aead.config import Config

    cfg = Config.from_dict({
        "experiment": "e2e", "output_dir": str(tmp_path_factory.mktemp("runs")),
        "data": {"root": str(tiny_dataset), "image_size": 64, "val_split": 0.25},
        "model": {"name": "memae", "channels": [8, 16, 32, 32], "latent_dim": 16,
                  "memory_size": 10},
        "train": {"epochs": 2, "batch_size": 8, "device": "cpu"},
    })
    return train(cfg)


def test_training_writes_artifacts(trained_run):
    for name in ["best.pt", "history.csv", "config.json"]:
        assert (trained_run / name).exists()


def test_evaluate_report(trained_run):
    m = evaluate(trained_run / "best.pt", device="cpu")
    assert 0.0 <= m["image_auroc"] <= 1.0 and 0.0 <= m["pixel_auroc"] <= 1.0
    assert set(m["per_defect_auroc"]) == {"scratch", "crack", "dent", "stain"}
    for f in ["metrics.json", "scores_roc.png", "examples.png"]:
        assert (trained_run / "eval" / f).exists()


def test_onnx_export_matches_pytorch(trained_run, tiny_dataset):
    paths = export(trained_run / "best.pt")  # raises if ONNX and PyTorch disagree
    img = next((tiny_dataset / "test" / "crack").glob("*.png"))
    torch_pred = Predictor.load(trained_run / "best.pt").predict_file(img)
    onnx_pred = Predictor.load(paths["onnx"]).predict_file(img)
    assert abs(torch_pred.score - onnx_pred.score) < 1e-4
    assert onnx_pred.anomaly_map.shape == (64, 64)


def test_onnx_runtime_path_does_not_import_torch(trained_run):
    export(trained_run / "best.pt")
    code = ("import sys; from aead.inference import Predictor; import aead.api; "
            f"Predictor.load(r'{trained_run / 'export' / 'model.onnx'}'); "
            "assert 'torch' not in sys.modules, 'torch was imported'")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_rest_api(trained_run, tiny_dataset):
    model = export(trained_run / "best.pt")["onnx"]
    img = next((tiny_dataset / "test" / "scratch").glob("*.png"))
    with TestClient(create_app(str(model))) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/model").json()["backend"] == "onnxruntime"
        r = client.post("/predict?heatmap=true",
                        files={"file": (img.name, img.read_bytes(), "image/png")})
        assert r.status_code == 200
        body = r.json()
        assert body["label"] in {"OK", "ANOMALY"} and "heatmap_png_base64" in body
        bad = client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")})
        assert bad.status_code == 400


def test_cli_predict(trained_run, tiny_dataset, tmp_path):
    from aead.cli import main

    out = tmp_path / "pred"
    main(["predict", "--model", str(trained_run / "best.pt"),
          "--source", str(tiny_dataset / "test" / "dent"), "--out", str(out)])
    assert (out / "predictions.csv").exists()
    assert len(list(out.glob("*_heatmap.png"))) == 3
    heat = cv2.imread(str(next(out.glob("*_heatmap.png"))))
    assert heat is not None and heat.shape == (64, 64, 3) and np.ptp(heat) > 0
