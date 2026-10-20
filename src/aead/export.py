"""Export the full detector (autoencoder + scoring head) for deployment.

The exported graph maps an image batch straight to (anomaly_map, score, reconstruction),
so serving needs only ONNX Runtime / OpenVINO plus NumPy — no PyTorch.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from aead.engine import load_checkpoint

log = logging.getLogger("aead")


def export(checkpoint: str | Path, out_dir: str | Path | None = None,
           formats: tuple[str, ...] = ("onnx",), opset: int = 17) -> dict[str, Path]:
    detector, cfg, calib = load_checkpoint(checkpoint, "cpu")
    out_dir = Path(out_dir or Path(checkpoint).parent / "export")
    out_dir.mkdir(parents=True, exist_ok=True)
    size = cfg.data.image_size
    dummy = torch.rand(1, 3, size, size)

    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        detector, dummy, str(onnx_path), opset_version=opset, dynamo=False,
        input_names=["image"], output_names=["anomaly_map", "score", "reconstruction"],
        dynamic_axes={n: {0: "batch"} for n in ["image", "anomaly_map", "score",
                                                 "reconstruction"]},
    )
    meta = {"image_size": size, "model": cfg.model.name, "calibration": calib,
            "input": "RGB float32 NCHW in [0, 1]", "config": cfg.to_dict()}
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2))
    outputs = {"onnx": onnx_path, "metadata": out_dir / "model.json"}
    _verify_onnx(detector, onnx_path, dummy)

    if "openvino" in formats:
        try:
            import openvino as ov
        except ImportError as e:
            raise SystemExit("OpenVINO export needs: pip install openvino") from e
        ov_model = ov.convert_model(str(onnx_path))
        ov.save_model(ov_model, str(out_dir / "model.xml"))
        outputs["openvino"] = out_dir / "model.xml"
        log.info("OpenVINO IR written to %s", out_dir / "model.xml")
    return outputs


def _verify_onnx(detector, onnx_path: Path, dummy: torch.Tensor) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed; skipping numerical check")
        return
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    batch = torch.cat([dummy, torch.rand_like(dummy)])  # also checks the dynamic batch axis
    onnx_map, onnx_score, _ = sess.run(None, {"image": batch.numpy()})
    with torch.no_grad():
        ref_map, ref_score, _ = detector(batch)
    err = float(np.abs(onnx_score - ref_score.numpy()).max())
    map_err = float(np.abs(onnx_map - ref_map.numpy()).max())
    log.info("ONNX export verified: max |Δscore| = %.2e, max |Δmap| = %.2e", err, map_err)
    if err > 1e-3:
        raise RuntimeError(f"ONNX output mismatch (max score error {err:.3e})")
