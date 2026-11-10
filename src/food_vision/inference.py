"""Torch-free inference with ONNX Runtime (what the API and Docker image use).

Preprocessing reimplements the eval transform from `data.build_transforms`
(Resize shorter side -> CenterCrop -> scale to [0, 1] -> normalise) with PIL + NumPy,
following torchvision's exact size/crop arithmetic so outputs match training-time
preprocessing (verified in tests).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessSpec:
    """Eval preprocessing parameters (stored in the ONNX file's metadata)."""

    image_size: int
    resize_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


def preprocess(image: Image.Image, spec: PreprocessSpec) -> np.ndarray:
    """PIL image -> float32 array (1, 3, H, W), matching the torchvision eval transform."""
    img = image.convert("RGB")
    w, h = img.size
    # torchvision Resize(int): shorter side -> resize_size, longer side scaled with int() truncation.
    if w <= h:
        new_w, new_h = spec.resize_size, int(spec.resize_size * h / w)
    else:
        new_w, new_h = int(spec.resize_size * w / h), spec.resize_size
    img = img.resize((new_w, new_h), Image.BILINEAR)
    # torchvision CenterCrop offsets: int(round((dim - crop) / 2)).
    top = int(round((new_h - spec.image_size) / 2.0))
    left = int(round((new_w - spec.image_size) / 2.0))
    img = img.crop((left, top, left + spec.image_size, top + spec.image_size))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(spec.mean, dtype=np.float32)) / np.array(spec.std, dtype=np.float32)
    return x.transpose(2, 0, 1)[None].copy()


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class OnnxClassifier:
    """An exported food classifier: ONNX session + class names + preprocessing from metadata."""

    def __init__(self, path: str | Path, providers: list[str] | None = None, threads: int | None = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.path = Path(path)
        self.session = ort.InferenceSession(str(path), opts, providers=providers or ["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        try:
            self.class_names: list[str] = json.loads(meta["class_names"])
            pre = json.loads(meta["preprocess"])
        except KeyError as e:
            raise ValueError(f"{path} lacks food-vision metadata ({e}); export it with scripts/export_onnx.py") from e
        self.spec = PreprocessSpec(pre["image_size"], pre["resize_size"], tuple(pre["mean"]), tuple(pre["std"]))
        self.model_name = meta.get("arch", self.path.stem)
        self.input_name = self.session.get_inputs()[0].name

    def logits(self, batch: np.ndarray) -> np.ndarray:
        """Raw logits for a preprocessed batch (N, 3, H, W)."""
        return self.session.run(None, {self.input_name: batch.astype(np.float32, copy=False)})[0]

    def predict(self, image: Image.Image, top_k: int = 3) -> list[tuple[str, float]]:
        """Top-k (class name, probability) for one PIL image."""
        probs = softmax(self.logits(preprocess(image, self.spec)))[0]
        top = np.argsort(-probs)[:top_k]
        return [(self.class_names[i], float(probs[i])) for i in top]
