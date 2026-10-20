# Lightweight inference image: ONNX Runtime + FastAPI, no PyTorch (~400 MB).
#
#   aead export --checkpoint runs/memae/best.pt          # creates runs/memae/export/
#   docker build -t aead-api --build-arg MODEL_DIR=runs/memae/export .
#   docker run -p 8000:8000 aead-api
FROM python:3.11-slim

ARG MODEL_DIR=models
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AEAD_MODEL=/app/model/model.onnx

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"

COPY ${MODEL_DIR}/model.onnx ${MODEL_DIR}/model.json /app/model/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "aead.api:app", "--host", "0.0.0.0", "--port", "8000"]
