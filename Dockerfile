# food-vision inference API: ONNX Runtime + FastAPI, no PyTorch.
# Build (after `python scripts/export_onnx.py --model vit`; optionally `python scripts/export_yolo.py`
# to enable POST /detect):
#   docker build -t food-vision .
#   docker run --rm -p 8000:8000 food-vision

# Stage 1: pick the model files. The YOLO detector is optional, and COPY can't skip a missing
# file, so it is resolved here and only the selected files reach the final image.
FROM python:3.12-slim AS models
ARG MODEL=models/vit.onnx
ARG DETECTOR=models/yolo11n.onnx
COPY models/ /src/models/
RUN mkdir /out && cp "/src/${MODEL}" /out/model.onnx \
    && if [ -f "/src/${DETECTOR}" ]; then cp "/src/${DETECTOR}" /out/detector.onnx; \
       else echo "No detector at ${DETECTOR}: POST /detect will return 503"; fi

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so code changes don't invalidate this layer.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-deps . && rm -rf src

# Exported models chosen in stage 1 (--build-arg MODEL=models/cnn.onnx also works).
COPY --from=models /out/ /app/
ENV FOOD_VISION_MODEL=/app/model.onnx \
    FOOD_VISION_DETECTOR=/app/detector.onnx

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "food_vision.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
