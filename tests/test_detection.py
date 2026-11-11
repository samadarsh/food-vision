"""Multi-food detection: geometry, NMS, the ONNX detector, detect+classify and POST /detect."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from food_vision.api.app import app
from food_vision.detection import (
    OnnxYoloDetector, crop_box, decode, detect_and_classify, draw_dishes, iou_matrix, letterbox, nms, unletterbox,
)
from food_vision.inference import OnnxClassifier

from .conftest import CLASSES, jpeg_bytes


def _image(w: int = 128, h: int = 64) -> Image.Image:
    return Image.open(io.BytesIO(jpeg_bytes((w, h))))


# --------------------------------------------------------------------------- geometry


def test_letterbox_matches_ultralytics_arithmetic() -> None:
    x, scale, pad = letterbox(_image(128, 64), (64, 64))
    assert x.shape == (1, 3, 64, 64) and x.dtype == np.float32
    assert scale == 0.5 and pad == (0, 16)                 # 128x64 -> 64x32, padded 16 top/bottom
    assert np.allclose(x[0, :, :16], 114 / 255)            # grey padding rows
    assert np.allclose(x[0, :, 48:], 114 / 255)


@pytest.mark.parametrize("size,pad", [
    ((100, 67), (0, 10)),   # new 64x43, dh = 10.5: round(10.4) = 10
    ((100, 64), (0, 11)),   # new 64x41, dh = 11.5: round(11.4) = 11 (plain round() would give 12)
    ((64, 100), (11, 0)),   # same horizontally
])
def test_letterbox_odd_padding_rounds_like_ultralytics(size, pad) -> None:
    assert letterbox(_image(*size), (64, 64))[2] == pad


def test_unletterbox_inverts_letterbox_and_clips() -> None:
    boxes = np.array([[12.0, 12.0, 28.0, 28.0], [0.0, 16.0, 64.0, 48.0]])
    out = unletterbox(boxes, 0.5, (0, 16), (128, 64))
    np.testing.assert_allclose(out, [[24, 0, 56, 24], [0, 0, 128, 64]])  # y=-8 clipped to 0


def test_iou_and_nms() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [20, 20, 30, 30]], dtype=np.float32)
    assert np.isclose(iou_matrix(boxes[:1], boxes[1:2])[0, 0], 90 / 110)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), iou_thr=0.5)
    assert keep.tolist() == [0, 2]


def test_decode_is_class_aware() -> None:
    out = np.zeros((1, 6, 2), np.float32)                  # 2 classes, 2 identical boxes
    out[0, :4] = [[10, 10], [10, 10], [8, 8], [8, 8]]
    out[0, 4, 0], out[0, 5, 1] = 0.9, 0.8                  # different classes -> both survive
    boxes, scores, cls = decode(out, conf=0.25, iou=0.5)
    assert sorted(cls.tolist()) == [0, 1] and len(boxes) == 2
    np.testing.assert_allclose(boxes[0], [6, 6, 14, 14])   # cxcywh -> xyxy


def test_decode_empty() -> None:
    boxes, scores, cls = decode(np.zeros((1, 6, 3), np.float32), conf=0.25, iou=0.5)
    assert boxes.shape == (0, 4) and len(scores) == len(cls) == 0


def test_crop_box_adds_context_and_clips() -> None:
    crop = crop_box(Image.new("RGB", (100, 100)), (0, 0, 50, 50), pad_frac=0.1)
    assert crop.size == (55, 55)


# --------------------------------------------------------------------------- detector + pipeline


def test_detector_reads_metadata_and_filters_food(tiny_yolo) -> None:
    det = OnnxYoloDetector(tiny_yolo)
    assert det.imgsz == (64, 64) and det.names[1] == "bowl" and det.model_name == "tiny-test-detector"
    food = det.detect(_image(128, 64))                     # person removed, bowl duplicate NMS'd, pizza < conf
    assert [(d.label, round(d.score, 2)) for d in food] == [("bowl", 0.8)]
    np.testing.assert_allclose(food[0].box, (24, 0, 56, 24))
    everything = det.detect(_image(128, 64), food_only=False)
    assert [d.label for d in everything] == ["person", "bowl"]
    assert [d.label for d in det.detect(_image(128, 64), conf=0.05)] == ["bowl", "pizza"]


def test_detector_rejects_non_yolo_onnx(tiny_onnx) -> None:
    with pytest.raises(ValueError, match="YOLO metadata"):
        OnnxYoloDetector(tiny_onnx)


def test_detect_and_classify(tiny_yolo, tiny_onnx) -> None:
    det, clf = OnnxYoloDetector(tiny_yolo), OnnxClassifier(tiny_onnx)
    dishes = detect_and_classify(_image(128, 64), det, clf, top_k=2)
    assert len(dishes) == 1 and dishes[0].detection.label == "bowl"
    assert len(dishes[0].predictions) == 2 and dishes[0].predictions[0][0] in CLASSES


def test_detect_and_classify_falls_back_to_whole_image(tiny_yolo, tiny_onnx) -> None:
    det, clf = OnnxYoloDetector(tiny_yolo), OnnxClassifier(tiny_onnx)
    img = _image(128, 64)
    dishes = detect_and_classify(img, det, clf, conf=0.95)  # nothing passes the threshold
    assert len(dishes) == 1 and dishes[0].detection is None and dishes[0].box == (0, 0, 128, 64)
    assert dishes[0].predictions == clf.predict(img, top_k=3)  # identical to single-image prediction
    assert detect_and_classify(img, det, clf, conf=0.95, fallback_whole_image=False) == []


def test_detect_and_classify_skips_tiny_boxes(tiny_yolo, tiny_onnx) -> None:
    det, clf = OnnxYoloDetector(tiny_yolo), OnnxClassifier(tiny_onnx)
    dishes = detect_and_classify(_image(128, 64), det, clf, min_size=40)  # bowl box is 32x24
    assert len(dishes) == 1 and dishes[0].detection is None


def test_draw_dishes_returns_annotated_copy(tiny_yolo, tiny_onnx) -> None:
    img = _image(128, 64).convert("RGB")
    dishes = detect_and_classify(img, OnnxYoloDetector(tiny_yolo), OnnxClassifier(tiny_onnx))
    out = draw_dishes(img, dishes)
    assert out.size == img.size and not np.array_equal(np.asarray(out), np.asarray(img))


# --------------------------------------------------------------------------- API


@pytest.fixture
def client(tiny_onnx, tiny_yolo, monkeypatch) -> TestClient:
    monkeypatch.setenv("FOOD_VISION_MODEL", str(tiny_onnx))
    monkeypatch.setenv("FOOD_VISION_DETECTOR", str(tiny_yolo))
    monkeypatch.setenv("FOOD_VISION_MAX_UPLOAD_MB", "0.05")
    with TestClient(app) as c:
        yield c


def _upload(data: bytes, content_type: str = "image/jpeg") -> dict:
    return {"file": ("meal.jpg", data, content_type)}


def test_detect_endpoint(client) -> None:
    r = client.post("/detect", files=_upload(jpeg_bytes((128, 64))))
    assert r.status_code == 200
    body = r.json()
    assert body["num_detections"] == 1 and len(body["dishes"]) == 1
    dish = body["dishes"][0]
    assert dish["source"] == "detection" and dish["detector_label"] == "bowl"
    assert dish["box"] == {"x1": 24.0, "y1": 0.0, "x2": 56.0, "y2": 24.0}
    assert [p["rank"] for p in dish["predictions"]] == [1, 2, 3]
    assert all(p["nutrition"]["calories"] > 0 for p in dish["predictions"])
    assert (body["image_width"], body["image_height"]) == (128, 64)
    assert body["detector"] == "tiny-test-detector" and "approximate" in body["nutrition_disclaimer"].lower()


def test_detect_applies_exif_orientation(client) -> None:
    """A phone photo stored sideways (EXIF orientation 6) is rotated upright before inference."""
    img = Image.open(io.BytesIO(jpeg_bytes((128, 64))))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    body = client.post("/detect?conf=0.95", files=_upload(buf.getvalue())).json()
    assert (body["image_width"], body["image_height"]) == (64, 128)
    assert body["dishes"][0]["box"] == {"x1": 0.0, "y1": 0.0, "x2": 64.0, "y2": 128.0}


def test_detect_endpoint_params_and_fallback(client) -> None:
    two = client.post("/detect?conf=0.05&top_k=1", files=_upload(jpeg_bytes((128, 64)))).json()
    assert [d["detector_label"] for d in two["dishes"]] == ["bowl", "pizza"]
    assert all(len(d["predictions"]) == 1 for d in two["dishes"])
    none = client.post("/detect?conf=0.95", files=_upload(jpeg_bytes((128, 64)))).json()
    assert none["num_detections"] == 0 and none["dishes"][0]["source"] == "whole_image"
    assert none["dishes"][0]["detector_label"] is None


@pytest.mark.parametrize("query", ["conf=0", "conf=1.5", "top_k=0"])
def test_detect_invalid_params_422(client, query) -> None:
    assert client.post(f"/detect?{query}", files=_upload(jpeg_bytes())).status_code == 422


def test_detect_reuses_upload_validation(client) -> None:
    assert client.post("/detect", files=_upload(b"hello", "text/plain")).status_code == 415
    assert client.post("/detect", files=_upload(b"not an image")).status_code == 400
    assert client.post("/detect", files=_upload(b"\xff" * 60_000)).status_code == 413


def test_predict_unchanged_when_detector_loaded(client) -> None:
    r = client.post("/predict", files=_upload(jpeg_bytes()))
    assert r.status_code == 200 and len(r.json()["predictions"]) == 3


def test_detect_without_detector_is_503(tiny_onnx, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FOOD_VISION_MODEL", str(tiny_onnx))
    monkeypatch.setenv("FOOD_VISION_DETECTOR", str(tmp_path / "missing.onnx"))
    with TestClient(app) as c:
        assert c.post("/detect", files=_upload(jpeg_bytes())).status_code == 503
        assert c.post("/predict", files=_upload(jpeg_bytes())).status_code == 200  # classifier still works
        assert c.get("/health").json()["status"] == "ok"


@pytest.mark.skipif(not Path("models/yolo11n.onnx").exists(), reason="export with scripts/export_yolo.py")
def test_real_detector_smoke() -> None:
    """The exported YOLO loads and returns well-formed boxes inside the image (no accuracy claim)."""
    det = OnnxYoloDetector("models/yolo11n.onnx")
    img = Image.new("RGB", (512, 384), (200, 180, 150))
    for d in det.detect(img, conf=0.01, food_only=False):
        x1, y1, x2, y2 = d.box
        assert 0 <= x1 <= x2 <= 512 and 0 <= y1 <= y2 <= 384 and 0 < d.score <= 1
