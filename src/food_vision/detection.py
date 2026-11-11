"""Multi-food detection: YOLO (ONNX) finds food regions, the existing classifier labels each one.

Food-101 has no bounding boxes, so the detector is a COCO-pretrained YOLO exported by
`scripts/export_yolo.py`. COCO's food classes (pizza, donut, cake, sandwich, ...) plus
`bowl` (curries, ramen, salads) act as food-region proposals; every region is cropped and
classified into the project's dish classes by `inference.OnnxClassifier`, then joined with
nutrition by the API.

Torch-free like `inference.py`: letterbox and NMS follow Ultralytics' arithmetic in
NumPy/PIL (agreement with Ultralytics is measured by scripts/export_yolo.py).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from food_vision.inference import OnnxClassifier

# COCO classes treated as food-region proposals: prepared dishes plus "bowl" (curries, ramen,
# salads, rice). Raw produce (banana, apple, orange, broccoli, carrot) is excluded by default: the
# dish classifier has no class for it, so garnish crops would be forced into a wrong dish label.
FOOD_LABELS: tuple[str, ...] = ("bowl", "sandwich", "hot dog", "pizza", "donut", "cake")
PAD_VALUE = 114  # Ultralytics letterbox fill (grey)
MAX_WH = 7680    # Ultralytics class offset for class-aware NMS


@dataclass(frozen=True)
class Detection:
    """One detected region in original-image pixel coordinates (x1, y1, x2, y2)."""

    box: tuple[float, float, float, float]
    score: float
    class_id: int
    label: str


@dataclass(frozen=True)
class DishResult:
    """A food region and the classifier's top-k dishes for its crop.

    `detection` is None when nothing was detected and the whole image was classified instead.
    """

    box: tuple[float, float, float, float]
    detection: Detection | None
    predictions: list[tuple[str, float]]


# --------------------------------------------------------------------------- geometry


def letterbox(image: Image.Image, size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to `size` (h, w), as Ultralytics' LetterBox(auto=False).

    Returns (float32 array (1, 3, h, w) in [0, 1], scale, (left, top) padding).
    """
    img = image.convert("RGB")
    w0, h0 = img.size
    h, w = size
    r = min(h / h0, w / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    if (new_w, new_h) != (w0, h0):
        img = img.resize((new_w, new_h), Image.BILINEAR)
    dw, dh = (w - new_w) / 2, (h - new_h) / 2
    left, top = int(round(dw - 0.1)), int(round(dh - 0.1))
    canvas = Image.new("RGB", (w, h), (PAD_VALUE,) * 3)
    canvas.paste(img, (left, top))
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1)[None].copy(), r, (left, top)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between xyxy boxes a (N, 4) and b (M, 4)."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy non-maximum suppression; returns kept indices, highest score first."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.array(keep, dtype=np.int64)


def decode(output: np.ndarray, conf: float, iou: float,
           max_det: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw YOLO output (1, 4 + nc, anchors) -> (xyxy boxes, scores, class ids) after class-aware NMS.

    Boxes are in letterboxed input pixels. Mirrors Ultralytics' non_max_suppression with
    multi_label=False and agnostic=False.
    """
    pred = output[0].T  # (anchors, 4 + nc)
    cls_scores = pred[:, 4:]
    class_ids = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(len(pred)), class_ids]
    m = scores > conf
    if not m.any():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    cxcywh, scores, class_ids = pred[m, :4], scores[m], class_ids[m]
    xyxy = np.concatenate([cxcywh[:, :2] - cxcywh[:, 2:] / 2, cxcywh[:, :2] + cxcywh[:, 2:] / 2], axis=1)
    keep = nms(xyxy + class_ids[:, None] * MAX_WH, scores, iou)[:max_det]  # offset => per-class NMS
    return xyxy[keep], scores[keep], class_ids[keep]


def unletterbox(boxes: np.ndarray, scale: float, pad: tuple[int, int], image_size: tuple[int, int]) -> np.ndarray:
    """Map letterboxed xyxy boxes back to original image pixels (w, h), clipped to the image."""
    out = boxes.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / scale
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, image_size[0])
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, image_size[1])
    return out


# --------------------------------------------------------------------------- detector


class OnnxYoloDetector:
    """A YOLO detector exported to ONNX by Ultralytics (raw output, no built-in NMS)."""

    def __init__(self, path: str | Path, providers: list[str] | None = None, threads: int | None = None,
                 food_labels: tuple[str, ...] = FOOD_LABELS):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.path = Path(path)
        self.session = ort.InferenceSession(str(path), opts, providers=providers or ["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        try:
            names = ast.literal_eval(meta["names"])
            imgsz = ast.literal_eval(meta["imgsz"])
        except KeyError as e:
            raise ValueError(f"{path} lacks YOLO metadata ({e}); export it with scripts/export_yolo.py") from e
        if meta.get("task", "detect") != "detect" or meta.get("end2end", "False") == "True":
            raise ValueError(f"{path}: expected a detection model with raw (non-end2end) output")
        self.names: dict[int, str] = {int(k): v for k, v in names.items()}
        self.imgsz: tuple[int, int] = (int(imgsz[0]), int(imgsz[1]))
        self.food_ids = {i for i, n in self.names.items() if n in food_labels}
        self.model_name = meta.get("description", self.path.stem).split(" trained")[0]
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image: Image.Image, conf: float = 0.25, iou: float = 0.7, food_only: bool = True,
               max_det: int = 300) -> list[Detection]:
        """Detections in original-image pixels, highest score first."""
        x, scale, pad = letterbox(image, self.imgsz)
        output = self.session.run(None, {self.input_name: x})[0]
        boxes, scores, class_ids = decode(output, conf, iou, max_det)
        boxes = unletterbox(boxes, scale, pad, image.size)
        dets = [Detection(tuple(float(v) for v in b), float(s), int(c), self.names.get(int(c), str(c)))
                for b, s, c in zip(boxes, scores, class_ids)]
        return [d for d in dets if d.class_id in self.food_ids] if food_only else dets


# --------------------------------------------------------------------------- detect + classify


def crop_box(image: Image.Image, box: tuple[float, float, float, float], pad_frac: float = 0.05) -> Image.Image:
    """Crop a box with a little context (pad_frac of its size on each side), clipped to the image."""
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    w, h = image.size
    return image.crop((max(0, int(x1 - px)), max(0, int(y1 - py)), min(w, int(round(x2 + px))),
                       min(h, int(round(y2 + py)))))


def detect_and_classify(image: Image.Image, detector: OnnxYoloDetector, classifier: OnnxClassifier,
                        top_k: int = 3, conf: float = 0.25, iou: float = 0.7, min_size: int = 16,
                        fallback_whole_image: bool = True) -> list[DishResult]:
    """Detect food regions and classify each crop with the dish classifier.

    Boxes smaller than `min_size` px on either side are skipped. If no food region is found
    and `fallback_whole_image` is set, the whole image is classified as a single dish
    (matching /predict's behaviour for close-up photos).
    """
    image = image.convert("RGB")
    results = []
    for det in detector.detect(image, conf=conf, iou=iou, food_only=True):
        x1, y1, x2, y2 = det.box
        if x2 - x1 < min_size or y2 - y1 < min_size:
            continue
        results.append(DishResult(det.box, det, classifier.predict(crop_box(image, det.box), top_k=top_k)))
    if not results and fallback_whole_image:
        w, h = image.size
        results.append(DishResult((0.0, 0.0, float(w), float(h)), None, classifier.predict(image, top_k=top_k)))
    return results


def draw_dishes(image: Image.Image, dishes: list[DishResult]) -> Image.Image:
    """Copy of `image` with each dish's box and top-1 label drawn on it."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    width = max(2, round(min(img.size) / 200))
    font = ImageFont.load_default(size=max(12, round(min(img.size) / 30)))
    colors = ["#E4572E", "#17BEBB", "#FFC914", "#76B041", "#8172B3", "#2E86AB"]
    for i, d in enumerate(dishes):
        color = colors[i % len(colors)]
        label, p = d.predictions[0]
        text = f"{label} {p:.0%}" + ("" if d.detection else " (whole image)")
        draw.rectangle(d.box, outline=color, width=width)
        ty = d.box[1] - font.size - 4 if d.box[1] > font.size + 4 else d.box[1] + width
        text_w = draw.textlength(text, font=font)
        tx = max(0, min(d.box[0] + width, img.width - text_w - 2))  # keep the label inside the image
        bbox = draw.textbbox((tx, ty), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((tx, ty), text, fill="black", font=font)
    return img
