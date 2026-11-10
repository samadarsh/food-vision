"""Evaluation and error analysis on saved test predictions (logits + labels)."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Predictions:
    """Test-set outputs of one model, as saved by `metrics.save_test_outputs`."""

    name: str
    logits: np.ndarray       # (N, C)
    labels: np.ndarray       # (N,)
    class_names: list[str]

    @classmethod
    def load(cls, name: str, path: Path) -> Predictions:
        d = np.load(path)
        return cls(name, d["logits"], d["labels"], [str(c) for c in d["class_names"]])

    @property
    def probs(self) -> np.ndarray:
        z = self.logits - self.logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    @property
    def preds(self) -> np.ndarray:
        return self.logits.argmax(axis=1)

    @property
    def correct(self) -> np.ndarray:
        return self.preds == self.labels


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    """cm[i, j] = number of images of true class i predicted as class j."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)
    return cm


def per_class_report(cm: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """Precision, recall, F1 and support per class, plus macro and weighted averages."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    df = pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support},
                      index=pd.Index(class_names, name="class_name"))
    w = support / support.sum()
    df.loc["macro avg"] = [precision.mean(), recall.mean(), f1.mean(), support.sum()]
    df.loc["weighted avg"] = [(precision * w).sum(), (recall * w).sum(), (f1 * w).sum(), support.sum()]
    df["support"] = df["support"].astype(int)
    return df


def most_confused_pairs(cm: np.ndarray, class_names: list[str], k: int = 10) -> pd.DataFrame:
    """Class pairs with the most errors in *either* direction (A->B plus B->A)."""
    rows = []
    n = len(class_names)
    for i in range(n):
        for j in range(i + 1, n):
            ab, ba = int(cm[i, j]), int(cm[j, i])
            if ab + ba:
                rows.append({"class_a": class_names[i], "class_b": class_names[j],
                             "a_as_b": ab, "b_as_a": ba, "total": ab + ba})
    df = pd.DataFrame(rows, columns=["class_a", "class_b", "a_as_b", "b_as_a", "total"])
    return df.sort_values(["total", "class_a"], ascending=[False, True]).head(k).reset_index(drop=True)


def worst_errors(p: Predictions, k: int = 16) -> pd.DataFrame:
    """Most *confidently* wrong predictions: misclassified images with the highest predicted prob."""
    probs = p.probs
    idx = np.flatnonzero(~p.correct)
    df = pd.DataFrame({
        "index": idx,
        "true": [p.class_names[i] for i in p.labels[idx]],
        "pred": [p.class_names[i] for i in p.preds[idx]],
        "pred_prob": probs[idx, p.preds[idx]],
        "true_prob": probs[idx, p.labels[idx]],
    })
    return df.sort_values("pred_prob", ascending=False).head(k).reset_index(drop=True)


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """Exact (binomial) McNemar test on paired correctness of two models on the same images.

    Only discordant pairs matter: b01 = A right & B wrong, b10 = A wrong & B right.
    Two-sided p-value under H0: both directions equally likely.
    """
    b01 = int((correct_a & ~correct_b).sum())
    b10 = int((~correct_a & correct_b).sum())
    n, k = b01 + b10, min(b01, b10)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return {"only_a_correct": b01, "only_b_correct": b10, "p_value": p}
