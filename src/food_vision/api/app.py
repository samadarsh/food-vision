"""FastAPI service: upload a food photo, get the top-k dishes with approximate nutrition.

Run locally:
    uvicorn food_vision.api.app:app --port 8000
Environment variables:
    FOOD_VISION_MODEL         path to an ONNX file from scripts/export_onnx.py (default models/vit.onnx)
    FOOD_VISION_MAX_UPLOAD_MB upload size limit in MB (default 10)
    FOOD_VISION_THREADS       ONNX Runtime intra-op threads (default: runtime decides)
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
from PIL import Image, UnidentifiedImageError

from food_vision.api.schemas import ClassInfo, HealthResponse, NutritionOut, Prediction, PredictResponse
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model and nutrition table once at startup."""
    path = os.environ.get("FOOD_VISION_MODEL", "models/vit.onnx")
    threads = int(os.environ["FOOD_VISION_THREADS"]) if os.environ.get("FOOD_VISION_THREADS") else None
    app.state.classifier = OnnxClassifier(path, threads=threads)
    app.state.nutrition = default_nutrition()
    app.state.max_bytes = int(float(os.environ.get("FOOD_VISION_MAX_UPLOAD_MB", "10")) * 1024 * 1024)
    if missing := app.state.nutrition.missing(app.state.classifier.class_names):
        log.warning("No nutrition entry for: %s", missing)
    log.info("Loaded %s (%d classes)", path, len(app.state.classifier.class_names))
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
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise HTTPException(400, f"Could not decode image: {type(e).__name__}") from e

    t0 = time.perf_counter()
    top = clf.predict(image, top_k=min(top_k, len(clf.class_names)))
    elapsed = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        predictions=[
            Prediction(rank=i + 1, label=label, display_name=display_name(label),
                       confidence=round(p, 6), nutrition=_nutrition(table, label))
            for i, (label, p) in enumerate(top)
        ],
        nutrition_disclaimer=table.disclaimer,
        nutrition_units=table.units,
        model=clf.model_name,
        inference_ms=round(elapsed, 2),
    )
