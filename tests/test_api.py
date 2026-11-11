"""FastAPI service, run against a tiny real ONNX model (no trained weights needed)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from food_vision.api.app import _min_confidence, app, is_confident

from .conftest import CLASSES, jpeg_bytes


@pytest.fixture
def client(tiny_onnx, monkeypatch) -> TestClient:
    monkeypatch.setenv("FOOD_VISION_MODEL", str(tiny_onnx))
    monkeypatch.setenv("FOOD_VISION_MAX_UPLOAD_MB", "0.05")  # ~51 KB, to test the size limit
    with TestClient(app) as c:  # runs the lifespan (model loading)
        yield c


def _upload(data: bytes, content_type: str = "image/jpeg") -> dict:
    return {"file": ("food.jpg", data, content_type)}


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": "tiny-test-model", "num_classes": len(CLASSES)}


def test_classes_include_nutrition(client) -> None:
    body = client.get("/classes").json()
    assert [c["label"] for c in body] == CLASSES
    assert all(c["nutrition"]["calories"] > 0 for c in body)
    assert body[0]["display_name"] == "Chicken curry"


def test_predict_returns_sorted_top3_with_nutrition(client) -> None:
    r = client.post("/predict", files=_upload(jpeg_bytes()))
    assert r.status_code == 200
    body = r.json()
    preds = body["predictions"]
    assert [p["rank"] for p in preds] == [1, 2, 3]
    confs = [p["confidence"] for p in preds]
    assert confs == sorted(confs, reverse=True) and abs(sum(confs) - 1) < 1e-4  # all 3 classes
    assert {p["label"] for p in preds} == set(CLASSES)
    assert all(p["nutrition"]["serving"] for p in preds)
    assert "approximate" in body["nutrition_disclaimer"].lower()
    assert body["model"] == "tiny-test-model" and body["inference_ms"] >= 0


def test_predict_top_k(client) -> None:
    assert len(client.post("/predict?top_k=1", files=_upload(jpeg_bytes())).json()["predictions"]) == 1


def test_predict_accepts_png_with_alpha(client) -> None:
    r = client.post("/predict", files=_upload(jpeg_bytes((120, 90), "RGBA"), "image/png"))
    assert r.status_code == 200


@pytest.mark.parametrize("query", ["top_k=0", "top_k=11", "top_k=abc"])
def test_invalid_top_k_is_422(client, query) -> None:
    assert client.post(f"/predict?{query}", files=_upload(jpeg_bytes())).status_code == 422


def test_missing_file_is_422(client) -> None:
    assert client.post("/predict").status_code == 422


def test_unsupported_content_type_is_415(client) -> None:
    assert client.post("/predict", files=_upload(b"hello", "text/plain")).status_code == 415


def test_non_image_bytes_is_400(client) -> None:
    r = client.post("/predict", files=_upload(b"definitely not a jpeg"))
    assert r.status_code == 400 and "decode" in r.json()["detail"]


def test_empty_upload_is_400(client) -> None:
    assert client.post("/predict", files=_upload(b"")).status_code == 400


def test_oversized_upload_is_413(client) -> None:
    assert client.post("/predict", files=_upload(b"\xff" * 60_000)).status_code == 413


def _client_with_floor(tiny_onnx, monkeypatch, floor: str) -> TestClient:
    monkeypatch.setenv("FOOD_VISION_MODEL", str(tiny_onnx))
    monkeypatch.setenv("FOOD_VISION_MIN_CONFIDENCE", floor)
    return TestClient(app)


def test_is_confident_uses_top1_and_includes_the_floor() -> None:
    assert is_confident([("pizza", 0.5), ("samosa", 0.3)], 0.5)  # "at least": equal counts
    assert not is_confident([("pizza", 0.4999), ("samosa", 0.4)], 0.5)
    assert is_confident([("pizza", 0.6), ("samosa", 0.1)], 0.5)  # only top-1 matters
    assert not is_confident([], 0.0)


def test_predict_confident_default_floor_is_half(client) -> None:
    body = client.post("/predict", files=_upload(jpeg_bytes())).json()
    assert body["confident"] is (body["predictions"][0]["confidence"] >= 0.5)


def test_predict_confident_follows_env_floor(tiny_onnx, monkeypatch) -> None:
    with _client_with_floor(tiny_onnx, monkeypatch, "0") as c:
        top1 = c.post("/predict", files=_upload(jpeg_bytes())).json()["predictions"][0]["confidence"]
    assert 0 < top1 < 1
    for floor, expected in [(top1 - 0.01, True), (top1 + 0.01, False)]:  # same image, floor either side of top-1
        with _client_with_floor(tiny_onnx, monkeypatch, str(floor)) as c:
            body = c.post("/predict", files=_upload(jpeg_bytes())).json()
        assert body["confident"] is expected and body["predictions"][0]["confidence"] == top1


@pytest.mark.parametrize("value, expected", [(None, 0.5), ("", 0.5), ("0.8", 0.8), ("0", 0.0), ("1", 1.0)])
def test_confidence_floor_default_and_env(monkeypatch, value, expected) -> None:
    if value is None:
        monkeypatch.delenv("FOOD_VISION_MIN_CONFIDENCE", raising=False)
    else:
        monkeypatch.setenv("FOOD_VISION_MIN_CONFIDENCE", value)
    assert _min_confidence() == expected


@pytest.mark.parametrize("bad", ["1.5", "-0.1", "high", "nan"])
def test_invalid_confidence_floor_fails_at_startup(tiny_onnx, monkeypatch, bad) -> None:
    with pytest.raises(ValueError, match="FOOD_VISION_MIN_CONFIDENCE"):
        with _client_with_floor(tiny_onnx, monkeypatch, bad):
            pass
