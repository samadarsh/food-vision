"""LR schedule, early stopping and config validation."""

from __future__ import annotations

import pytest
import torch
import yaml

from food_vision.config import load_config
from food_vision.engine import EarlyStopping, warmup_cosine


def test_warmup_then_cosine() -> None:
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    sched = warmup_cosine(opt, warmup_steps=10, total_steps=110, min_ratio=0.01)
    lrs = []
    for _ in range(110):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)  # linear warmup
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))                # monotone decay
    assert lrs[60] == pytest.approx(0.505, abs=0.01)                       # cosine midpoint
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01)                # floor


def test_early_stopping() -> None:
    es = EarlyStopping(patience=2)
    assert es.step(0.5) and es.step(0.6)
    assert not es.step(0.6) and not es.should_stop  # equal is not an improvement
    assert not es.step(0.55) and es.should_stop
    assert es.best == 0.6


def test_default_config_loads() -> None:
    cfg = load_config()
    assert len(cfg.data.class_names) == 20 and {"cnn", "vit"} <= set(cfg.classifiers)


def test_unknown_config_keys_are_rejected(tmp_path) -> None:
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({"data": {"classes": ["a", "b"]}, "trian": {}}))
    with pytest.raises(ValueError, match="unknown top-level keys"):
        load_config(tmp_path / "c.yaml")
