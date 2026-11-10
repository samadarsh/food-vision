"""Classifier construction (torchvision CNNs or timm ViTs) and checkpoint I/O."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch import nn

from food_vision.data import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class Preprocess:
    """What a model expects as input; stored in checkpoints so inference matches training."""

    image_size: int = 224
    mean: tuple[float, ...] = IMAGENET_MEAN
    std: tuple[float, ...] = IMAGENET_STD


def is_torchvision_arch(arch: str) -> bool:
    """True if `arch` is a torchvision classification model name (else assumed timm)."""
    return arch in torchvision.models.list_models(module=torchvision.models)


def _replace_torchvision_head(model: nn.Module, num_classes: int) -> None:
    """Swap the final Linear layer for a fresh one with `num_classes` outputs."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):  # ResNet family
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return
    head = getattr(model, "classifier", None)  # EfficientNet / MobileNet / ConvNeXt ...
    if isinstance(head, nn.Sequential) and isinstance(head[-1], nn.Linear):
        head[-1] = nn.Linear(head[-1].in_features, num_classes)
        return
    raise ValueError(f"Don't know how to replace the head of {type(model).__name__}")


def build_classifier(arch: str, num_classes: int, pretrained: bool = True) -> tuple[nn.Module, Preprocess]:
    """Create an ImageNet-pretrained classifier with a new `num_classes` head."""
    if is_torchvision_arch(arch):
        model = torchvision.models.get_model(arch, weights="DEFAULT" if pretrained else None)
        _replace_torchvision_head(model, num_classes)
        return model, Preprocess()

    import timm

    model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
    dc = timm.data.resolve_model_data_config(model)
    return model, Preprocess(image_size=dc["input_size"][-1], mean=tuple(dc["mean"]), std=tuple(dc["std"]))


def count_parameters(model: nn.Module) -> int:
    """Total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path: Path, model: nn.Module, *, arch: str, class_names: list[str],
                    preprocess: Preprocess, **meta: Any) -> None:
    """Save weights (on CPU) plus everything needed to rebuild the model for inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "arch": arch,
            "class_names": class_names,
            "preprocess": asdict(preprocess),
            **meta,
        },
        path,
    )


def load_classifier(path: Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a classifier from `save_checkpoint` output. Returns (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model, _ = build_classifier(ckpt["arch"], len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    ckpt["preprocess"] = Preprocess(**{k: tuple(v) if isinstance(v, list) else v
                                       for k, v in ckpt["preprocess"].items()})
    return model.to(device).eval(), ckpt
