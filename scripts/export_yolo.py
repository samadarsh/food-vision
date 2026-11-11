"""Export a COCO-pretrained YOLO detector to ONNX and verify the torch-free pipeline against Ultralytics.

The served detector (`detection.OnnxYoloDetector`) re-implements letterboxing and NMS in
NumPy/PIL. This script measures how closely it reproduces Ultralytics on test images
(written to <reports_dir>/deploy/yolo_verify.json):
  * pass/fail: vs Ultralytics running the *same* ONNX file (identical 640x640 letterbox input),
    which isolates the re-implemented pre/post-processing
  * informational: vs Ultralytics' PyTorch weights, which use rectangular inference
    (e.g. 640x480 input) and therefore differ slightly from any static 640x640 export
  * metrics: matched detections (same class, IoU >= --match-iou) in both directions,
    mean IoU and max |score difference| of matched pairs

Requires `pip install ultralytics` (dev-time only; the API and Docker image don't need it).
Note: Ultralytics and its pretrained weights are AGPL-3.0 licensed.

Usage:
    python scripts/export_yolo.py [--weights yolo11n.pt] [--out models/yolo11n.onnx] [--n-images 200]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from food_vision.config import DEFAULT_CONFIG, load_config
from food_vision.data import food101_dir, get_splits
from food_vision.detection import OnnxYoloDetector, iou_matrix


def match(a_boxes: np.ndarray, a_cls: np.ndarray, b_boxes: np.ndarray, b_cls: np.ndarray,
          min_iou: float) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching of same-class boxes by IoU (highest first)."""
    if not len(a_boxes) or not len(b_boxes):
        return []
    ious = iou_matrix(a_boxes, b_boxes) * (a_cls[:, None] == b_cls[None, :])
    pairs, used_a, used_b = [], set(), set()
    for flat in np.argsort(-ious, axis=None):
        i, j = np.unravel_index(flat, ious.shape)
        if ious[i, j] < min_iou:
            break
        if i not in used_a and j not in used_b:
            pairs.append((int(i), int(j), float(ious[i, j])))
            used_a.add(i)
            used_b.add(j)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--weights", default="yolo11n.pt", help="Ultralytics weights (downloaded if missing)")
    parser.add_argument("--out", default="models/yolo11n.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--n-images", type=int, default=200, help="test images used for verification")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.9)
    parser.add_argument("--min-match-rate", type=float, default=0.95, help="pass threshold (both directions)")
    args = parser.parse_args()

    from ultralytics import YOLO

    cfg = load_config(args.config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report_dir = cfg.data.reports_dir / "deploy"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Download the weights next to the ONNX file, then export (raw output, static 640x640, no NMS).
    weights = out.parent / args.weights
    with tempfile.TemporaryDirectory() as tmp:
        tmp_weights = Path(tmp) / args.weights
        if weights.exists():
            shutil.copy(weights, tmp_weights)
        model = YOLO(str(tmp_weights))  # Ultralytics downloads known weights to this path
        exported = Path(model.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=False))
        shutil.move(exported, out)
        if not weights.exists():
            shutil.copy(model.ckpt_path, weights)
    print(f"Exported {out} ({out.stat().st_size / 1e6:.1f} MB)")

    detector = OnnxYoloDetector(out)
    references = {"same_onnx": YOLO(str(out), task="detect"), "pytorch_rect": YOLO(str(weights))}
    test = get_splits(cfg)["test"]
    images_dir = food101_dir(cfg.data.root) / "images"
    sample = random.Random(cfg.seed).sample(test, min(args.n_images, len(test)))

    stats = {k: {"reference_detections": 0, "onnx_detections": 0, "matched": 0, "ious": [], "diffs": []}
             for k in references}
    food_images = 0
    for s in tqdm(sample, desc="Ultralytics vs ONNX pipeline", leave=False):
        path = images_dir / s.rel_path
        with Image.open(path) as im:
            dets = detector.detect(im, conf=args.conf, iou=args.iou, food_only=False)
        ob = np.array([d.box for d in dets], dtype=np.float32).reshape(-1, 4)
        oc = np.array([d.class_id for d in dets], dtype=int)
        os_ = np.array([d.score for d in dets], dtype=np.float32)
        food_images += any(d.class_id in detector.food_ids for d in dets)
        for key, ref in references.items():
            r = ref.predict(str(path), imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]
            rb, rc, rs = (r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                          r.boxes.conf.cpu().numpy())
            pairs = match(rb, rc, ob, oc, args.match_iou)
            st = stats[key]
            st["reference_detections"] += len(rb)
            st["onnx_detections"] += len(ob)
            st["matched"] += len(pairs)
            st["ious"] += [p[2] for p in pairs]
            st["diffs"] += [abs(float(rs[i]) - float(os_[j])) for i, j, _ in pairs]

    comparisons = {}
    for key, st in stats.items():
        comparisons[key] = {
            "reference_detections": st["reference_detections"], "onnx_detections": st["onnx_detections"],
            "matched": st["matched"],
            "match_rate_vs_reference": (st["matched"] / st["reference_detections"]
                                        if st["reference_detections"] else 1.0),
            "match_rate_vs_onnx": st["matched"] / st["onnx_detections"] if st["onnx_detections"] else 1.0,
            "mean_iou_matched": float(np.mean(st["ious"])) if st["ious"] else None,
            "max_abs_score_diff_matched": float(np.max(st["diffs"])) if st["diffs"] else None,
        }
    main_cmp = comparisons["same_onnx"]
    report = {
        "weights": args.weights, "onnx_path": str(out), "onnx_size_mb": round(out.stat().st_size / 1e6, 1),
        "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou, "match_iou": args.match_iou,
        "num_images": len(sample), "source": "random sample of the test split (seeded)",
        "comparisons": comparisons,
        "pass_criterion": f"same_onnx match rate >= {args.min_match_rate} in both directions",
        "images_with_food_detection": food_images,
        "food_labels": sorted(detector.names[i] for i in detector.food_ids),
        "license_note": "Ultralytics YOLO weights and code are AGPL-3.0",
    }
    report["passed"] = bool(min(main_cmp["match_rate_vs_reference"], main_cmp["match_rate_vs_onnx"])
                            >= args.min_match_rate)
    (report_dir / "yolo_verify.json").write_text(json.dumps(report, indent=2))

    print()  # tqdm (leave=False) clears its line; start the summary on a fresh one
    for key, label in [("same_onnx", "vs Ultralytics, same ONNX (pass/fail)"),
                       ("pytorch_rect", "vs Ultralytics, PyTorch rect inference (info)")]:
        c = comparisons[key]
        print(f"{label}: reference {c['reference_detections']}, ours {c['onnx_detections']}, matched {c['matched']} "
              f"-> {c['match_rate_vs_reference']:.2%} / {c['match_rate_vs_onnx']:.2%}, "
              f"mean IoU {c['mean_iou_matched']:.4f}, max |score diff| {c['max_abs_score_diff_matched']:.4f}")
    print(f"  images with >= 1 food-region detection: {food_images}/{len(sample)}")
    print("PASSED" if report["passed"] else "FAILED", f"-> {report_dir / 'yolo_verify.json'}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
