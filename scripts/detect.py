"""Detect and classify every dish in one or more photos (YOLO regions -> dish classifier).

Saves an annotated copy of each image and prints the dishes with approximate nutrition.
Torch-free: uses the same ONNX components as the API's POST /detect.

Usage:
    python scripts/detect.py IMAGE [IMAGE ...] [--detector models/yolo11n.onnx]
                             [--classifier models/vit.onnx] [--out reports/detect] [--conf 0.25]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps

from food_vision.detection import OnnxYoloDetector, detect_and_classify, draw_dishes
from food_vision.inference import OnnxClassifier
from food_vision.nutrition import default_nutrition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--detector", default="models/yolo11n.onnx")
    parser.add_argument("--classifier", default="models/vit.onnx")
    parser.add_argument("--out", type=Path, default=Path("reports/detect"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    detector, classifier = OnnxYoloDetector(args.detector), OnnxClassifier(args.classifier)
    table = default_nutrition()
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        with Image.open(path) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")  # honour phone-photo rotation
        dishes = detect_and_classify(image, detector, classifier, top_k=args.top_k, conf=args.conf)
        out = args.out / f"{path.stem}_detected.jpg"
        draw_dishes(image, dishes).save(out, quality=90)
        print(f"{path} -> {out}")
        for i, d in enumerate(dishes, 1):
            label, p = d.predictions[0]
            n = table.lookup(label)
            where = (f"{d.detection.label} {d.detection.score:.2f} at "
                     f"({', '.join(f'{v:.0f}' for v in d.box)})") if d.detection else "whole image (no detection)"
            kcal = f", ~{n.calories:.0f} kcal per {n.serving}" if n else ""
            print(f"  dish {i}: {label} ({p:.0%}) | region: {where}{kcal}")
    print(f"Note: {table.disclaimer}")


if __name__ == "__main__":
    main()
