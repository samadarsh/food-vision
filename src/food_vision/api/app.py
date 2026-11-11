"""FastAPI service: upload a food photo, get the top-k dishes with approximate nutrition.

Run locally:
    uvicorn food_vision.api.app:app --port 8000
Environment variables:
    FOOD_VISION_MODEL         path to an ONNX file from scripts/export_onnx.py (default models/vit.onnx)
    FOOD_VISION_MAX_UPLOAD_MB upload size limit in MB (default 10)
    FOOD_VISION_THREADS       ONNX Runtime intra-op threads (default: runtime decides)
    FOOD_VISION_DETECTOR      YOLO ONNX file from scripts/export_yolo.py (default models/yolo11n.onnx);
                              optional: if missing, POST /detect returns 503 and /predict is unaffected
    FOOD_VISION_MIN_CONFIDENCE top-1 probability below which a result is flagged `confident: false`
                              (default 0.5); flags non-food photos and dishes outside the classes
"""

from __future__ import annotations

import io
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from food_vision.api.schemas import (
    Box, ClassInfo, DetectedDish, DetectResponse, HealthResponse, NutritionOut, Prediction, PredictResponse,
)
from food_vision.detection import OnnxYoloDetector, detect_and_classify
from food_vision.inference import OnnxClassifier
from food_vision.nutrition import NutritionTable, default_nutrition

log = logging.getLogger("food_vision.api")
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif", "application/octet-stream"}


def display_name(label: str) -> str:
    """'chicken_curry' -> 'Chicken curry'."""
    return label.replace("_", " ").capitalize()


def _nutrition(table: NutritionTable, label: str) -> NutritionOut | None:
    n = table.lookup(label)
    return NutritionOut(**n.to_dict()) if n else None


def _predictions(top: list[tuple[str, float]], table: NutritionTable) -> list[Prediction]:
    """Classifier top-k -> ranked API predictions with nutrition."""
    return [Prediction(rank=i + 1, label=label, display_name=display_name(label),
                       confidence=round(p, 6), nutrition=_nutrition(table, label))
            for i, (label, p) in enumerate(top)]


def _min_confidence() -> float:
    """Confidence floor from FOOD_VISION_MIN_CONFIDENCE (default 0.5); must be a probability."""
    raw = os.environ.get("FOOD_VISION_MIN_CONFIDENCE") or "0.5"  # unset or empty -> default
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not 0.0 <= value <= 1.0:  # also rejects NaN
        raise ValueError(f"FOOD_VISION_MIN_CONFIDENCE must be a number between 0 and 1, got {raw!r}")
    return value


def is_confident(top: list[tuple[str, float]], min_confidence: float) -> bool:
    """True when the top-1 probability reaches the confidence floor."""
    return bool(top) and top[0][1] >= min_confidence


def _read_image(file: UploadFile, max_bytes: int) -> Image.Image:
    """Validate an upload (type, size, decodability) and return the decoded image."""
    if file.content_type and file.content_type.lower() not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported content type {file.content_type!r}; upload an image")
    data = file.file.read(max_bytes + 1)
    if not data:
        raise HTTPException(400, "Empty upload")
    if len(data) > max_bytes:
        raise HTTPException(413, f"File larger than {max_bytes // (1024 * 1024)} MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        # Phone photos store rotation in EXIF; apply it so the model sees the photo upright
        # and /detect boxes are in the coordinates the user sees.
        image = ImageOps.exif_transpose(image)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise HTTPException(400, f"Could not decode image: {type(e).__name__}") from e
    return image


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model and nutrition table once at startup."""
    path = os.environ.get("FOOD_VISION_MODEL", "models/vit.onnx")
    threads = int(os.environ["FOOD_VISION_THREADS"]) if os.environ.get("FOOD_VISION_THREADS") else None
    app.state.classifier = OnnxClassifier(path, threads=threads)
    app.state.nutrition = default_nutrition()
    app.state.max_bytes = int(float(os.environ.get("FOOD_VISION_MAX_UPLOAD_MB", "10")) * 1024 * 1024)
    app.state.min_confidence = _min_confidence()
    if missing := app.state.nutrition.missing(app.state.classifier.class_names):
        log.warning("No nutrition entry for: %s", missing)
    log.info("Loaded %s (%d classes)", path, len(app.state.classifier.class_names))
    det_path = os.environ.get("FOOD_VISION_DETECTOR", "models/yolo11n.onnx")
    app.state.detector = OnnxYoloDetector(det_path, threads=threads) if os.path.exists(det_path) else None
    log.info("Detector: %s", det_path if app.state.detector else f"not found at {det_path}; /detect disabled")
    yield


app = FastAPI(
    title="food-vision",
    description="Food recognition with approximate nutritional insights. "
                "Nutrition values are rough per-serving references, not measurements.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    clf: OnnxClassifier = request.app.state.classifier
    return HealthResponse(status="ok", model=clf.model_name, num_classes=len(clf.class_names))


@app.get("/classes", response_model=list[ClassInfo])
def classes(request: Request) -> list[ClassInfo]:
    clf: OnnxClassifier = request.app.state.classifier
    table: NutritionTable = request.app.state.nutrition
    return [ClassInfo(label=c, display_name=display_name(c), nutrition=_nutrition(table, c))
            for c in clf.class_names]


@app.post("/predict", response_model=PredictResponse)
def predict(  # sync: FastAPI runs it in a worker thread, so CPU inference doesn't block the event loop
    request: Request,
    file: UploadFile = File(..., description="A food photo (JPEG, PNG, WebP, ...)"),
    top_k: int = Query(3, ge=1, le=10, description="Number of dishes to return"),
) -> PredictResponse:
    clf: OnnxClassifier = request.app.state.classifier
    table: NutritionTable = request.app.state.nutrition
    max_bytes: int = request.app.state.max_bytes

    image = _read_image(file, max_bytes)

    t0 = time.perf_counter()
    top = clf.predict(image, top_k=min(top_k, len(clf.class_names)))
    elapsed = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        predictions=_predictions(top, table),
        confident=is_confident(top, request.app.state.min_confidence),
        nutrition_disclaimer=table.disclaimer,
        nutrition_units=table.units,
        model=clf.model_name,
        inference_ms=round(elapsed, 2),
    )


@app.post("/detect", response_model=DetectResponse)
def detect(  # sync, like /predict: CPU inference runs in FastAPI's worker threads
    request: Request,
    file: UploadFile = File(..., description="A photo that may contain several dishes"),
    top_k: int = Query(3, ge=1, le=10, description="Dishes returned per detected region"),
    conf: float = Query(0.25, ge=0.01, le=0.99, description="Detector confidence threshold"),
) -> DetectResponse:
    """Find food regions with YOLO and classify each one; falls back to the whole image if none."""
    detector: OnnxYoloDetector | None = request.app.state.detector
    if detector is None:
        raise HTTPException(503, "Detector not loaded: export it with scripts/export_yolo.py "
                                 "(or set FOOD_VISION_DETECTOR)")
    clf: OnnxClassifier = request.app.state.classifier
    table: NutritionTable = request.app.state.nutrition
    image = _read_image(file, request.app.state.max_bytes)

    t0 = time.perf_counter()
    dishes = detect_and_classify(image, detector, clf, top_k=min(top_k, len(clf.class_names)), conf=conf)
    elapsed = (time.perf_counter() - t0) * 1000

    return DetectResponse(
        dishes=[
            DetectedDish(
                box=Box(x1=round(d.box[0], 1), y1=round(d.box[1], 1), x2=round(d.box[2], 1), y2=round(d.box[3], 1)),
                source="detection" if d.detection else "whole_image",
                detector_label=d.detection.label if d.detection else None,
                detector_confidence=round(d.detection.score, 6) if d.detection else None,
                predictions=_predictions(d.predictions, table),
                confident=is_confident(d.predictions, request.app.state.min_confidence),
            )
            for d in dishes
        ],
        num_detections=sum(d.detection is not None for d in dishes),
        image_width=image.width,
        image_height=image.height,
        nutrition_disclaimer=table.disclaimer,
        nutrition_units=table.units,
        detector=detector.model_name,
        model=clf.model_name,
        inference_ms=round(elapsed, 2),
    )
