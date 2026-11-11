"""Shared fixtures: a tiny fake Food-101 tree and a tiny ONNX classifier (no real data needed)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from food_vision.config import Config, load_config

CLASSES = ["chicken_curry", "pizza", "samosa"]


@pytest.fixture(scope="session")
def fake_food101(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """data root containing food-101/{images,meta} with 3 classes, 20 train + 8 test images each."""
    root = tmp_path_factory.mktemp("data")
    base = root / "food-101"
    (base / "meta").mkdir(parents=True)
    rng = random.Random(0)
    train, test = {}, {}
    for c in CLASSES + ["apple_pie"]:  # one extra class that the config doesn't use
        (base / "images" / c).mkdir(parents=True)
        train[c] = [f"{c}/{i}" for i in range(20)]
        test[c] = [f"{c}/{100 + i}" for i in range(8)]
        for i in train[c] + test[c]:
            size = (rng.choice([512, 384, 300]), rng.choice([512, 382, 260]))
            Image.new("RGB", size, tuple(rng.randrange(256) for _ in range(3))).save(base / "images" / f"{i}.jpg")
    (base / "meta" / "classes.txt").write_text("\n".join(sorted(train)))
    (base / "meta" / "train.json").write_text(json.dumps(train))
    (base / "meta" / "test.json").write_text(json.dumps(test))
    (base / ".extracted").touch()
    return root


@pytest.fixture
def make_config(fake_food101: Path, tmp_path: Path):
    """Factory: write a config pointing at the fake dataset, with optional data overrides."""

    def _make(**data_overrides) -> Config:
        data = {"root": str(fake_food101), "splits_dir": str(tmp_path / "splits"),
                "reports_dir": str(tmp_path / "reports"), "classes": list(CLASSES),
                "train_per_class": 10, "val_per_class": 4, "test_per_class": 5, "num_workers": 0}
        data.update(data_overrides)
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({"seed": 42, "data": data}))
        return load_config(path)

    return _make


@pytest.fixture(scope="session")
def tiny_onnx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real ONNX classifier (GlobalAveragePool -> Flatten -> Gemm) over CLASSES, with food-vision metadata."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.standard_normal((len(CLASSES), 3)).astype(np.float32), "W")
    b = numpy_helper.from_array(np.zeros(len(CLASSES), np.float32), "B")
    graph = helper.make_graph(
        [helper.make_node("GlobalAveragePool", ["image"], ["pooled"]),
         helper.make_node("Flatten", ["pooled"], ["flat"]),
         helper.make_node("Gemm", ["flat", "W", "B"], ["logits"], transB=1)],
        "tiny",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 32, 32])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", len(CLASSES)])],
        initializer=[w, b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    for k, v in {"class_names": json.dumps(CLASSES), "arch": "tiny-test-model",
                 "preprocess": json.dumps({"image_size": 32, "resize_size": 36,
                                           "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]})}.items():
        model.metadata_props.add(key=k, value=v)
    onnx.checker.check_model(model)
    path = tmp_path_factory.mktemp("models") / "tiny.onnx"
    onnx.save(model, path)
    return path


def jpeg_bytes(size: tuple[int, int] = (300, 200), mode: str = "RGB") -> bytes:
    """Encode a random image as JPEG (or PNG for modes JPEG can't store)."""
    import io

    arr = np.random.default_rng(1).integers(0, 256, (size[1], size[0], len(mode)), dtype=np.uint8)
    img = Image.fromarray(arr.squeeze())  # mode inferred from channel count
    buf = io.BytesIO()
    img.save(buf, format="PNG" if mode in ("RGBA", "P") else "JPEG")
    return buf.getvalue()


DETECTOR_NAMES = {0: "person", 1: "bowl", 2: "pizza"}


@pytest.fixture(scope="session")
def tiny_yolo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real ONNX 'detector' with Ultralytics-style raw output (1, 4 + nc, anchors) and metadata.

    Input 64x64; the output is constant (it ignores the pixels), with 4 anchors in letterbox pixels:
      0: person  cx=32 cy=32 w=20 h=20  score 0.9   (non-food)
      1: bowl    cx=20 cy=20 w=16 h=16  score 0.8
      2: bowl    cx=21 cy=20 w=16 h=16  score 0.6   (duplicate of 1 -> removed by NMS)
      3: pizza   cx=48 cy=48 w=20 h=20  score 0.1   (below the default conf 0.25)
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    out = np.zeros((1, 4 + len(DETECTOR_NAMES), 4), np.float32)
    for a, (box, cls, score) in enumerate([((32, 32, 20, 20), 0, 0.9), ((20, 20, 16, 16), 1, 0.8),
                                           ((21, 20, 16, 16), 1, 0.6), ((48, 48, 20, 20), 2, 0.1)]):
        out[0, :4, a] = box
        out[0, 4 + cls, a] = score
    zero = numpy_helper.from_array(np.zeros((), np.float32), "zero")
    const = numpy_helper.from_array(out, "const")
    graph = helper.make_graph(
        [helper.make_node("ReduceSum", ["images"], ["s"], keepdims=0),  # touch the input...
         helper.make_node("Mul", ["s", "zero"], ["z"]),                   # ...then zero it out
         helper.make_node("Add", ["const", "z"], ["output0"])],
        "tiny_yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(out.shape))],
        initializer=[zero, const],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    for k, v in {"names": repr(DETECTOR_NAMES), "imgsz": "[64, 64]", "task": "detect", "end2end": "False",
                 "description": "tiny-test-detector trained on nothing"}.items():
        model.metadata_props.add(key=k, value=v)
    onnx.checker.check_model(model)
    path = tmp_path_factory.mktemp("models") / "tiny_yolo.onnx"
    onnx.save(model, path)
    return path
