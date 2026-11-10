"""Classification metrics shared by every model so they are evaluated identically."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def topk_correct(logits: torch.Tensor, labels: torch.Tensor, ks: tuple[int, ...] = (1, 5)) -> dict[int, int]:
    """Number of samples whose true label is within the top-k predictions, for each k."""
    maxk = min(max(ks), logits.shape[1])
    top = logits.topk(maxk, dim=1).indices
    hit = top.eq(labels.view(-1, 1))
    return {k: int(hit[:, : min(k, maxk)].any(dim=1).sum()) for k in ks}


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, ks: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    """Top-k accuracy (0-1) as {'top1': ..., 'top5': ...}."""
    n = len(labels)
    return {f"top{k}": c / n for k, c in topk_correct(logits, labels, ks).items()}


def save_test_outputs(out_dir: Path, logits: torch.Tensor, labels: torch.Tensor,
                      class_names: list[str], extra: dict | None = None) -> dict:
    """Write test_predictions.npz (raw logits + labels, for Phase 3) and test_metrics.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "test_predictions.npz", logits=logits.float().numpy(),
                        labels=labels.numpy(), class_names=np.array(class_names))
    metrics = {**topk_accuracy(logits.float(), labels), "num_test_images": len(labels), **(extra or {})}
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
