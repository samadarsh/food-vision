"""Metrics and statistics used to compare the models."""

from __future__ import annotations

import numpy as np
import torch

from food_vision.evaluation import confusion_matrix, mcnemar_exact, most_confused_pairs, per_class_report
from food_vision.metrics import topk_accuracy


def test_topk_accuracy() -> None:
    logits = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.05, 0.15], [0.2, 0.3, 0.5]])
    labels = torch.tensor([1, 2, 0])
    acc = topk_accuracy(logits, labels, ks=(1, 2, 5))  # k > num_classes is clamped
    assert acc == {"top1": 1 / 3, "top2": 2 / 3, "top5": 1.0}


def test_confusion_matrix_and_report() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    preds = np.array([0, 1, 1, 1, 2, 0])
    cm = confusion_matrix(labels, preds, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    rep = per_class_report(cm, ["a", "b", "c"])
    assert np.isclose(rep.loc["b", "precision"], 2 / 3) and rep.loc["b", "recall"] == 1.0
    assert np.isclose(rep.loc["a", "f1"], 0.5)
    assert rep.loc["macro avg", "support"] == 6


def test_most_confused_pairs_counts_both_directions() -> None:
    cm = np.array([[5, 3, 0], [2, 5, 1], [0, 0, 5]])
    top = most_confused_pairs(cm, ["a", "b", "c"], k=1).iloc[0]
    assert (top.class_a, top.class_b, top.a_as_b, top.b_as_a, top.total) == ("a", "b", 3, 2, 5)


def test_mcnemar() -> None:
    same = np.array([True, False, True])
    assert mcnemar_exact(same, same)["p_value"] == 1.0
    a = np.array([True] * 20 + [False] * 0)
    b = np.array([False] * 20)
    r = mcnemar_exact(a, b)
    assert r["only_a_correct"] == 20 and r["only_b_correct"] == 0
    assert np.isclose(r["p_value"], 2 * 0.5**20)
