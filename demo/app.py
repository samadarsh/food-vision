"""Gradio demo (Hugging Face Space): upload a food photo, see every dish boxed and labelled.

Torch-free: the same ONNX code the API's POST /detect uses (`food_vision.detection` and
`food_vision.inference`), plus the bundled nutrition table. On startup it downloads the two
ONNX models from the GitHub release.

Run locally:
    pip install -r demo/requirements.txt
    python demo/app.py
Environment variables:
    FOOD_VISION_MODEL_DIR       where the models are stored (default: models/ next to this file)
    FOOD_VISION_MIN_CONFIDENCE  top-1 probability below which a dish is marked "low confidence" (default 0.5)
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps

from food_vision.detection import OnnxYoloDetector, detect_and_classify, draw_dishes
from food_vision.inference import OnnxClassifier
from food_vision.nutrition import default_nutrition

RELEASE = "https://github.com/samadarsh/food-vision/releases/download/v0.1.0"
MODEL_DIR = Path(os.environ.get("FOOD_VISION_MODEL_DIR") or Path(__file__).parent / "models")
MIN_CONFIDENCE = float(os.environ.get("FOOD_VISION_MIN_CONFIDENCE") or 0.5)
HEADERS = ["Dish", "Confidence", "Found as", "Serving", "kcal", "Protein (g)", "Carbs (g)", "Fat (g)"]


def fetch(name: str) -> Path:
    """Download one release asset into MODEL_DIR unless it is already there."""
    dest = MODEL_DIR / name
    if not dest.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        print(f"Downloading {RELEASE}/{name}")
        try:
            with urllib.request.urlopen(f"{RELEASE}/{name}", timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            tmp.replace(dest)  # only a complete download gets the final name
        finally:
            tmp.unlink(missing_ok=True)
    return dest


CLASSIFIER = OnnxClassifier(fetch("vit.onnx"))
DETECTOR = OnnxYoloDetector(fetch("yolo11n.onnx"))
NUTRITION = default_nutrition()


def analyse(image: Image.Image | None, conf: float) -> tuple[Image.Image | None, list[list], str]:
    """Photo -> (annotated image, one table row per dish, summary with the nutrition disclaimer)."""
    if image is None:
        return None, [], "Upload a photo first."
    image = ImageOps.exif_transpose(image).convert("RGB")  # honour phone-photo rotation
    dishes = detect_and_classify(image, DETECTOR, CLASSIFIER, top_k=1, conf=conf)

    rows, unsure = [], 0
    for d in dishes:
        label, p = d.predictions[0]
        n = NUTRITION.lookup(label)
        low = p < MIN_CONFIDENCE
        unsure += low
        rows.append([
            label.replace("_", " ").capitalize() + (" (low confidence)" if low else ""),
            f"{p:.1%}",
            f"{d.detection.label} region ({d.detection.score:.2f})" if d.detection else "whole image",
            n.serving if n else "n/a",
            *([n.calories, n.protein, n.carbs, n.fat] if n else ["n/a"] * 4),
        ])

    found = (f"**{len(dishes)} food region(s) found.**" if dishes[0].detection
             else "**No separate food region found, so the whole photo was classified.**")
    if unsure:
        found += (f" {unsure} result(s) are below {MIN_CONFIDENCE:.0%} confidence: probably not one of the "
                  f"{len(CLASSIFIER.class_names)} dishes this model knows, or not food at all.")
    return draw_dishes(image, dishes), rows, f"{found}\n\n> **Nutrition disclaimer:** {NUTRITION.disclaimer}"


with gr.Blocks(title="food-vision") as demo:
    gr.Markdown(
        "# food-vision\n"
        "Upload a photo of a meal. A YOLO11n detector finds the food regions, a ViT-S/16 fine-tuned on "
        f"{len(CLASSIFIER.class_names)} Food-101 dishes names each one, and a lookup adds approximate nutrition "
        "per typical serving. Code and results: "
        "[github.com/samadarsh/food-vision](https://github.com/samadarsh/food-vision)."
    )
    with gr.Row():
        with gr.Column():
            photo = gr.Image(type="pil", label="Food photo", sources=["upload", "webcam", "clipboard"])
            conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Detector confidence threshold",
                             info="Lower finds more regions; higher keeps only the clearest ones.")
            run = gr.Button("Analyse", variant="primary")
        with gr.Column():
            annotated = gr.Image(type="pil", label="Detected dishes", interactive=False)
    table = gr.Dataframe(headers=HEADERS, label="Dishes (nutrition is approximate, per typical serving)",
                         interactive=False, wrap=True)
    summary = gr.Markdown()
    with gr.Accordion(f"The {len(CLASSIFIER.class_names)} dishes this model knows", open=False):
        gr.Markdown(", ".join(c.replace("_", " ") for c in CLASSIFIER.class_names))

    run.click(analyse, inputs=[photo, conf], outputs=[annotated, table, summary])
    photo.upload(analyse, inputs=[photo, conf], outputs=[annotated, table, summary])

if __name__ == "__main__":
    demo.launch()
