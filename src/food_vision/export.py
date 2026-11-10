"""Export trained classifiers to ONNX and verify them against PyTorch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def export_onnx(model: nn.Module, path: Path, *, image_size: int, resize_size: int,
                mean: tuple[float, ...], std: tuple[float, ...], class_names: list[str],
                arch: str, opset: int = 18) -> Path:
    """Export `model` (logits output, dynamic batch) to a single self-describing ONNX file.

    Class names and preprocessing are embedded as metadata so inference needs only the file.
    """
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.randn(2, 3, image_size, image_size)
    batch = torch.export.Dim("batch", min=1, max=1024)
    torch.onnx.export(
        model, (dummy,), str(path), input_names=["image"], output_names=["logits"],
        dynamic_shapes=({0: batch},), opset_version=opset, dynamo=True, external_data=False,
    )
    proto = onnx.load(str(path))
    meta = {
        "class_names": json.dumps(class_names),
        "preprocess": json.dumps({"image_size": image_size, "resize_size": resize_size,
                                  "mean": list(mean), "std": list(std)}),
        "arch": arch,
        "producer": "food-vision scripts/export_onnx.py",
    }
    del proto.metadata_props[:]
    for k, v in meta.items():
        proto.metadata_props.add(key=k, value=v)
    onnx.checker.check_model(proto)
    onnx.save(proto, str(path))
    return path


def compare_logits(torch_logits: np.ndarray, onnx_logits: np.ndarray, labels: np.ndarray) -> dict:
    """Agreement statistics between PyTorch and ONNX Runtime outputs on the same inputs."""
    diff = np.abs(torch_logits - onnx_logits)
    t_pred, o_pred = torch_logits.argmax(1), onnx_logits.argmax(1)
    return {
        "num_images": int(len(labels)),
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
        "top1_agreement": float((t_pred == o_pred).mean()),
        "torch_top1": float((t_pred == labels).mean()),
        "onnx_top1": float((o_pred == labels).mean()),
    }
