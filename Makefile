.PHONY: install data train-all evaluate export serve test lint format docker

MODEL ?= memae

install:            ## Install with training, serving and dev extras
	pip install -e ".[train,serve,dev]"

data:               ## Generate the synthetic inspection dataset
	aead synth --out data/synthetic

train-all: data     ## Train + evaluate + export all three models
	for m in conv_ae vae memae; do \
		aead train --config configs/$$m.yaml --evaluate && \
		aead export --checkpoint runs/$$m/best.pt; \
	done

evaluate:           ## Evaluate one model (MODEL=conv_ae|vae|memae)
	aead evaluate --checkpoint runs/$(MODEL)/best.pt

export:             ## Export one model to ONNX
	aead export --checkpoint runs/$(MODEL)/best.pt

serve:              ## Serve the exported ONNX model on :8000
	aead serve --model runs/$(MODEL)/export/model.onnx

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests

docker:             ## Build the inference image for MODEL
	docker build -t aead-api --build-arg MODEL_DIR=runs/$(MODEL)/export .
