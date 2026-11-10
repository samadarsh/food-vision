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
class Config:
    """Top-level project configuration."""

    data: DataConfig
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)  # sections added by later phases

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view (paths as strings), handy for logging."""
        d = asdict(self)
        d["data"] = {k: str(v) if isinstance(v, Path) else v for k, v in d["data"].items()}
        return d


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    """Load and validate a YAML config file."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    if "data" not in raw:
        raise ValueError(f"{path}: missing 'data' section")
    data = DataConfig(**raw.pop("data"))
    seed = int(raw.pop("seed", 42))
    return Config(data=data, seed=seed, extra=raw)
