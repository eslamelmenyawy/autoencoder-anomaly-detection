# Autoencoder Anomaly Detection for Visual Inspection

Unsupervised defect detection and localisation with Convolutional, Variational and Memory-augmented autoencoders, with ONNX/OpenVINO export and a FastAPI service.

> 🚧 Benchmark results and full documentation are being added.

```bash
pip install -e ".[train,serve,dev]"
aead synth --out data/synthetic
aead train --config configs/memae.yaml --evaluate
aead export --checkpoint runs/memae/best.pt
aead serve --model runs/memae/export/model.onnx
```
