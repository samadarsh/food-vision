"""Typed configuration loaded from a YAML file."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/default.yaml")


@dataclass
class DataConfig:
    """Dataset location, class subset and loader settings."""

    classes: list[str]
    root: Path = Path("data")
    splits_dir: Path = Path("data/splits")
    reports_dir: Path = Path("reports")
    train_per_class: int | None = 300
    val_per_class: int | None = 75
    test_per_class: int | None = 100
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    exclude: list[str] = field(default_factory=list)  # image ids like "dumplings/180290"

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.splits_dir = Path(self.splits_dir)
        self.reports_dir = Path(self.reports_dir)
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("Duplicate entries in data.classes")
        if len(self.classes) < 2:
            raise ValueError("data.classes needs at least 2 classes")
        for name in ("train_per_class", "val_per_class", "test_per_class"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"data.{name} must be >= 1 or null")

    @property
    def class_names(self) -> list[str]:
        """Classes in label order (sorted, so label ids are independent of YAML order)."""
        return sorted(self.classes)

    def split_fingerprint(self, seed: int) -> str:
        """Hash of every setting that affects which images land in which split."""
        payload = {
            "seed": seed,
            "classes": self.class_names,
            "train": self.train_per_class,
            "val": self.val_per_class,
            "test": self.test_per_class,
            "exclude": sorted(self.exclude),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class TrainConfig:
    """Optimisation settings shared by every trained model."""

    epochs: int = 15
    warmup_epochs: float = 1.0
    label_smoothing: float = 0.1
    grad_clip: float | None = 1.0
    patience: int = 4               # early stopping on val top-1 (epochs without improvement)
    amp: bool = True                # mixed precision on MPS/CUDA (ignored on CPU)
    device: str = "auto"            # auto | mps | cuda | cpu
    runs_dir: Path = Path("runs")

    def __post_init__(self) -> None:
        self.runs_dir = Path(self.runs_dir)
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("train.epochs and train.patience must be >= 1")


@dataclass
class ClassifierConfig:
    """A fine-tuned classifier: torchvision or timm architecture + its optimiser settings."""

    arch: str
    lr: float = 3e-4
    weight_decay: float = 0.05
    batch_size: int | None = None   # None -> data.batch_size


@dataclass
class ClipConfig:
    """Zero-shot CLIP (OpenCLIP) model and prompt templates."""

    arch: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    templates: list[str] = field(default_factory=lambda: ["a photo of {}, a type of food."])
    batch_size: int | None = None

    def __post_init__(self) -> None:
        if not self.templates or any("{}" not in t for t in self.templates):
            raise ValueError("clip.templates must be non-empty and each contain '{}'")


@dataclass
class Config:
    """Top-level project configuration."""

    data: DataConfig
    seed: int = 42
    train: TrainConfig = field(default_factory=TrainConfig)
    classifiers: dict[str, ClassifierConfig] = field(default_factory=dict)  # e.g. cnn, vit
    clip: ClipConfig = field(default_factory=ClipConfig)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view (paths as strings), handy for logging."""

        def clean(x: Any) -> Any:
            if isinstance(x, dict):
                return {k: clean(v) for k, v in x.items()}
            if isinstance(x, list):
                return [clean(v) for v in x]
            return str(x) if isinstance(x, Path) else x

        return clean(asdict(self))


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    """Load and validate a YAML config file (unknown keys raise a TypeError)."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    if "data" not in raw:
        raise ValueError(f"{path}: missing 'data' section")
    known = {"data", "seed", "train", "models"}
    if unknown := set(raw) - known:
        raise ValueError(f"{path}: unknown top-level keys {sorted(unknown)}")
    models = dict(raw.get("models") or {})
    clip = ClipConfig(**models.pop("clip", {}))
    return Config(
        data=DataConfig(**raw["data"]),
        seed=int(raw.get("seed", 42)),
        train=TrainConfig(**(raw.get("train") or {})),
        classifiers={k: ClassifierConfig(**v) for k, v in models.items()},
        clip=clip,
    )
