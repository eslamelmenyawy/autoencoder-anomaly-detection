# Training on your own dataset

No code changes are needed: arrange your images in the layout below and run one command.

## 1. Folder layout (MVTec AD style)

```
my_product/
├── train/good/                 ← normal images ONLY (≥ 200, ideally 300–1000)
├── test/good/                  ← normal images never used in training (≥ 30)
├── test/<defect_type>/         ← defective images, one folder per defect type (any name)
└── ground_truth/<defect_type>/<image_stem>_mask.png   ← OPTIONAL, white = defect
```

- Training uses **only** `train/good`. Defect images are used for evaluation only.
- Masks are optional. Without them you still get image AUROC, F1 and heatmaps; with them you also get pixel AUROC.
- PNG, JPG, BMP and TIFF are supported, including **16-bit** industrial-camera images.
- 10% of `train/good` is held out automatically to calibrate the decision threshold.

## 2. Install

```bash
git clone https://github.com/eslamelmenyawy/autoencoder-anomaly-detection.git
cd autoencoder-anomaly-detection
conda create -n aead python=3.11 -y && conda activate aead
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or /cpu
pip install -e ".[train,serve,dev]"
```

## 3. Train and evaluate

```bash
aead train --config configs/memae.yaml --data path/to/my_product --name my_product --evaluate
```

Outputs in `runs/my_product/`:

| File | Contents |
|---|---|
| `best.pt` | Trained model with its config and calibrated threshold |
| `history.csv` | Train/validation loss per epoch |
| `eval/metrics.json` | Image AUROC, AP, pixel AUROC, per-defect AUROC, F1, false alarms, latency |
| `eval/examples.png` | Input · reconstruction · heatmap · ground truth for each defect type |
| `eval/scores_roc.png` | Score distribution with the threshold, ROC per defect type |

Training only (no defect images yet): drop `--evaluate`.

## 4. Deploy

```bash
aead export --checkpoint runs/my_product/best.pt                       # ONNX (+ --format openvino)
aead predict --model runs/my_product/export/model.onnx --source new_images/ --out results/
aead serve --model runs/my_product/export/model.onnx                    # REST API on :8000
```

## 5. Tuning for real data

| Symptom | What to change |
|---|---|
| Parts appear at different positions or scales | Crop to the region of interest before training: an autoencoder flags *any* variation |
| Small defects are missed | Copy a config and set `image_size: 256` (must divide by 16); prefer `memae` or `vae` at this size |
| Thin scratches are missed | Lower `scoring.blur_sigma` to 2 and `scoring.topk_ratio` to 0.005 |
| Too many false alarms | Raise `scoring.threshold_percentile` (e.g. 99 → 99.5), add more varied normal images |
| Too many missed defects | Lower `scoring.threshold_percentile` (e.g. 99 → 98) |
| Reconstruction column in `examples.png` is flat grey | The model collapsed to the mean image: train longer (`--epochs 100`) or lower `train.lr` to 0.0005 |
| Colour defects (stains, discolouration) are missed | Lower `scoring.ssim_weight` toward 0.2 so the L2 colour residual counts more |

Any setting can live in your own YAML file; copy one from `configs/` and pass it with `--config`.
