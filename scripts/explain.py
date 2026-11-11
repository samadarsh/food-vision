"""Visualise what each model looks at: Grad-CAM (CNN) vs attention rollout (ViT).

The same test images go through both models; each row shows
    original | CNN Grad-CAM for its predicted class | ViT attention rollout.

Image sets (in <reports_dir>/explain/):
    correct.jpg   one image per chosen class that both models get right
    errors.jpg    the most confidently wrong images of each model + images all models miss

Usage:
    python scripts/explain.py [--classes samosa chicken_curry ...] [--n-errors 4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from food_vision.config import DEFAULT_CONFIG, load_config  # noqa: E402
from food_vision.data import FoodDataset, build_transforms, food101_dir, get_splits  # noqa: E402
from food_vision.evaluation import Predictions, worst_errors  # noqa: E402
from food_vision.explain import (  # noqa: E402
    AttentionRollout, GradCAM, artifact_token_check, denormalize, overlay,
)
from food_vision.models import load_classifier  # noqa: E402

DEFAULT_CLASSES = ["samosa", "chicken_curry", "fried_rice", "dumplings", "pizza", "sushi",
                   "steak", "ice_cream"]


def pick_correct(cnn: Predictions, vit: Predictions, classes: list[str]) -> list[int]:
    """For each class: the test image both models classify correctly with the highest joint confidence."""
    both = cnn.correct & vit.correct
    conf = cnn.probs.max(1) * vit.probs.max(1)
    picks = []
    for name in classes:
        c = cnn.class_names.index(name)
        idx = np.flatnonzero(both & (cnn.labels == c))
        if len(idx):
            picks.append(int(idx[np.argmax(conf[idx])]))
    return picks


def pick_errors(cnn: Predictions, vit: Predictions, clip: Predictions | None, n: int) -> list[int]:
    """Top-n confidently wrong for CNN and ViT, then images every model gets wrong (deduplicated)."""
    picks: list[int] = []
    for p in (cnn, vit):
        picks += [int(i) for i in worst_errors(p, k=n)["index"] if int(i) not in picks]
    wrong = ~cnn.correct & ~vit.correct & (~clip.correct if clip is not None else True)
    picks += [int(i) for i in np.flatnonzero(wrong) if int(i) not in picks][:n]
    return picks


def render(indices: list[int], cnn_ds: FoodDataset, vit_ds: FoodDataset, cam: GradCAM,
           rollout: AttentionRollout, class_names: list[str], cnn_pre, vit_pre,
           title: str, out: Path) -> None:
    """One row per image: original | Grad-CAM | rollout, with predictions in the titles."""
    fig, axes = plt.subplots(len(indices), 3, figsize=(8.4, 2.95 * len(indices)), squeeze=False)
    for row, i in enumerate(indices):
        x_cnn, label = cnn_ds[i]
        x_vit, _ = vit_ds[i]
        img = denormalize(x_cnn, cnn_pre.mean, cnn_pre.std)

        heat_c, pred_c, p_c = cam(x_cnn.unsqueeze(0))
        heat_v, pred_v, p_v = rollout(x_vit.unsqueeze(0))

        def mark(pred: int) -> str:
            return "✓" if pred == label else "✗"

        panels = [
            (img, f"true: {class_names[label]}"),
            (overlay(img, heat_c), f"CNN Grad-CAM {mark(pred_c)}\n{class_names[pred_c]} ({p_c:.0%})"),
            (overlay(img, heat_v), f"ViT rollout {mark(pred_v)}\n{class_names[pred_v]} ({p_v:.0%})"),
        ]
        for ax, (im, t) in zip(axes[row], panels):
            ax.imshow(im)
            ax.set_title(t, fontsize=8.5)
            ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))  # leave room for the title
    fig.savefig(out, dpi=100, pil_kwargs={"quality": 85, "optimize": True})
    plt.close(fig)
    print(f"  wrote {out} ({len(indices)} images)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--runs-dir")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--n-errors", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    runs = Path(args.runs_dir or cfg.train.runs_dir)
    out = cfg.data.reports_dir / "explain"
    out.mkdir(parents=True, exist_ok=True)
    names = cfg.data.class_names
    test = get_splits(cfg)["test"]
    images_dir = food101_dir(cfg.data.root) / "images"

    # CPU: a few dozen images, and autograd hooks behave identically everywhere.
    cnn_model, cnn_ckpt = load_classifier(runs / "cnn" / "best.pt", "cpu")
    vit_model, vit_ckpt = load_classifier(runs / "vit" / "best.pt", "cpu")
    cnn_pre, vit_pre = cnn_ckpt["preprocess"], vit_ckpt["preprocess"]
    # Same geometry (resize 256 -> centre crop 224) for both; only normalisation differs.
    cnn_ds = FoodDataset(test, images_dir, build_transforms(cnn_pre.image_size, False, cnn_pre.mean, cnn_pre.std))
    vit_ds = FoodDataset(test, images_dir, build_transforms(vit_pre.image_size, False, vit_pre.mean, vit_pre.std))

    cnn_p = Predictions.load("cnn", runs / "cnn" / "test_predictions.npz")
    vit_p = Predictions.load("vit", runs / "vit" / "test_predictions.npz")
    clip_path = runs / "clip" / "test_predictions.npz"
    clip_p = Predictions.load("clip", clip_path) if clip_path.exists() else None

    unknown = [c for c in args.classes if c not in names]
    if unknown:
        parser.error(f"--classes not in the config: {unknown}")

    cam, rollout = GradCAM(cnn_model), AttentionRollout(vit_model)
    render(pick_correct(cnn_p, vit_p, args.classes), cnn_ds, vit_ds, cam, rollout, names, cnn_pre,
           vit_pre, "Correct predictions: where each model looks", out / "correct.jpg")
    render(pick_errors(cnn_p, vit_p, clip_p, args.n_errors), cnn_ds, vit_ds, cam, rollout, names,
           cnn_pre, vit_pre, "Errors: confidently wrong / missed by every model", out / "errors.jpg")

    # Are rollout hot spots on background patches high-norm "artifact" tokens?
    idx = np.random.default_rng(cfg.seed).choice(len(vit_ds), 100, replace=False)
    checks = [artifact_token_check(rollout, [vit_ds[int(i)][0] for i in idx], layer=layer)
              for layer in (len(vit_model.blocks) // 2 - 1, 2 * len(vit_model.blocks) // 3, len(vit_model.blocks) - 1)]
    (out / "artifact_check.json").write_text(json.dumps(checks, indent=2))
    for c in checks:
        print(f"  block {c['layer']:>2}: max/median token norm {c['median_max_over_median_norm']:.1f}x, "
              f"rollout top-{c['top_k']} ∩ norm top-{c['top_k']} = {c['mean_topk_overlap']:.0%} "
              f"(chance {c['chance_topk_overlap']:.0%})")
    print(f"Outputs in {out}/")


if __name__ == "__main__":
    main()
