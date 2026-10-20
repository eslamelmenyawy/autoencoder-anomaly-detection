"""Command-line interface:  aead {synth,train,evaluate,export,predict,serve}."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from aead.data import IMAGE_EXTS


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")


def cmd_synth(a: argparse.Namespace) -> None:
    from aead.synthetic import generate_dataset

    root = generate_dataset(a.out, a.n_train, a.n_test_good, a.n_per_defect, a.size, a.seed)
    logging.getLogger("aead").info("Synthetic dataset written to %s", root)


def cmd_train(a: argparse.Namespace) -> None:
    from aead.config import Config
    from aead.engine import train

    cfg = Config.from_yaml(a.config)
    for key, value in [("data.root", a.data), ("train.epochs", a.epochs),
                       ("train.device", a.device), ("experiment", a.name)]:
        if value is not None:
            section, _, attr = key.rpartition(".")
            setattr(getattr(cfg, section) if section else cfg, attr, value)
    run_dir = train(cfg)
    if a.evaluate:
        from aead.evaluate import evaluate

        evaluate(run_dir / "best.pt", device=cfg.train.device)


def cmd_evaluate(a: argparse.Namespace) -> None:
    from aead.evaluate import evaluate

    metrics = evaluate(a.checkpoint, a.data, a.out, a.device)
    print(json.dumps({k: metrics[k] for k in ("image_auroc", "pixel_auroc", "image_ap")
                      if k in metrics}, indent=2))


def cmd_export(a: argparse.Namespace) -> None:
    from aead.export import export

    paths = export(a.checkpoint, a.out, tuple(a.format))
    for name, p in paths.items():
        print(f"{name:>9}: {p}")


def cmd_predict(a: argparse.Namespace) -> None:
    import cv2

    from aead.data import preprocess
    from aead.inference import Predictor
    from aead.visualize import heatmap_overlay

    predictor = Predictor.load(a.model, a.device)
    src = Path(a.source)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS) \
        if src.is_dir() else [src]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vmax = predictor.calibration.get("map_vis_max", 1.0)
    rows = []
    for f in files:
        pred = predictor.predict_file(f)
        rows.append({"file": str(f), **pred.to_dict()})
        print(f"{pred.to_dict()['label']:>7}  score={pred.score:.5f}  "
              f"{pred.latency_ms:6.1f} ms  {f}")
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        overlay = heatmap_overlay(preprocess(img, predictor.image_size), pred.anomaly_map, vmax)
        cv2.imwrite(str(out / f"{f.stem}_heatmap.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    with open(out / "predictions.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    n_anom = sum(r["is_anomaly"] for r in rows)
    print(f"\n{len(rows)} images | {n_anom} anomalous | results in {out}/")


def cmd_serve(a: argparse.Namespace) -> None:
    import uvicorn

    from aead.api import create_app

    uvicorn.run(create_app(a.model), host=a.host, port=a.port)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aead",
                                description="Autoencoder anomaly detection for visual inspection")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="Generate the synthetic inspection dataset")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--n-train", type=int, default=400)
    s.add_argument("--n-test-good", type=int, default=60)
    s.add_argument("--n-per-defect", type=int, default=25)
    s.add_argument("--size", type=int, default=256)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    t = sub.add_parser("train", help="Train a model from a YAML config")
    t.add_argument("--config", required=True)
    t.add_argument("--data", help="Override data.root")
    t.add_argument("--epochs", type=int, help="Override train.epochs")
    t.add_argument("--device", help="Override train.device (auto/cpu/cuda/mps)")
    t.add_argument("--name", help="Override experiment name (run folder)")
    t.add_argument("--evaluate", action="store_true", help="Evaluate on the test split after")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="Metrics + figures on the test split")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--data", help="Dataset root (default: the one used for training)")
    e.add_argument("--out", help="Report folder (default: <run>/eval)")
    e.add_argument("--device", default="auto")
    e.set_defaults(func=cmd_evaluate)

    x = sub.add_parser("export", help="Export to ONNX (and optionally OpenVINO)")
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--out", help="Output folder (default: <run>/export)")
    x.add_argument("--format", nargs="+", default=["onnx"], choices=["onnx", "openvino"])
    x.set_defaults(func=cmd_export)

    r = sub.add_parser("predict", help="Score an image or a folder and save heatmaps")
    r.add_argument("--model", required=True, help=".pt, .onnx or .xml")
    r.add_argument("--source", required=True, help="Image file or folder")
    r.add_argument("--out", default="predictions")
    r.add_argument("--device", default="cpu")
    r.set_defaults(func=cmd_predict)

    v = sub.add_parser("serve", help="Start the REST API")
    v.add_argument("--model", required=True)
    v.add_argument("--host", default="0.0.0.0")
    v.add_argument("--port", type=int, default=8000)
    v.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
