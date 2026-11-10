"""Evaluate and compare all models on the test split; error analysis; latency.

Reads runs/<model>/test_predictions.npz (written in Phase 2), so no re-inference is
needed for the metrics. Models are reloaded only for the latency benchmark.

Outputs (in <reports_dir>/eval/):
    comparison.csv / .md            model, params, top-1/5, macro-F1, latency (MPS + CPU)
    significance.csv                pairwise exact McNemar tests
    <model>_per_class.csv           precision / recall / F1 / support per class
    <model>_confusion.png           row-normalised confusion matrix
    <model>_confused_pairs.csv      most-confused class pairs (both directions)
    <model>_worst_errors.csv / .jpg most confidently wrong test images
    all_wrong.csv / .jpg            images every model gets wrong
    summary.md                      the above tables in one place (for the README)

Usage:
    python scripts/evaluate.py [--models cnn vit clip] [--skip-latency]
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from food_vision.benchmark import torch_latency  # noqa: E402
from food_vision.config import DEFAULT_CONFIG, Config, load_config  # noqa: E402
from food_vision.data import food101_dir, get_splits  # noqa: E402
from food_vision.device import get_device  # noqa: E402
from food_vision.evaluation import (  # noqa: E402
    Predictions, confusion_matrix, mcnemar_exact, most_confused_pairs, per_class_report, worst_errors,
)
from food_vision.models import count_parameters, load_classifier  # noqa: E402

DISPLAY = {"cnn": "EfficientNet-B0 (fine-tuned)", "vit": "ViT-S/16 (fine-tuned)",
           "clip": "CLIP ViT-B/32 (zero-shot)"}


def plot_confusion(cm: np.ndarray, names: list[str], title: str, out: Path) -> None:
    """Row-normalised confusion matrix with percentages in non-zero cells."""
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, f"{100 * norm[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def square_thumb(path: Path, size: int = 224) -> Image.Image:
    """Centre square crop, resized; matches what the models see (up to resize ratio)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        s = min(im.size)
        left, top = (im.width - s) // 2, (im.height - s) // 2
        return im.crop((left, top, left + s, top + s)).resize((size, size))


def image_grid(paths: list[Path], captions: list[str], out: Path, cols: int = 4, title: str = "") -> None:
    """Grid of thumbnails with a caption under each."""
    rows = max(1, int(np.ceil(len(paths) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3.1), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, p, c in zip(axes.flat, paths, captions):
        ax.imshow(square_thumb(p))
        ax.set_title(c, fontsize=7.5)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.45 / fig.get_figheight() if title else 0)))
    fig.savefig(out, dpi=100, pil_kwargs={"quality": 85, "optimize": True})
    plt.close(fig)


def latency_rows(keys: list[str], cfg: Config, runs: Path) -> dict[str, dict]:
    """Batch-1 fp32 forward latency on MPS/CUDA (if present) and CPU, plus parameter counts."""
    devices = [d for d in [get_device(cfg.train.device)] if d.type != "cpu"] + [torch.device("cpu")]
    out: dict[str, dict] = {}
    for key in keys:
        if key == "clip":
            from food_vision.clip_zero_shot import ZeroShotCLIP

            model = ZeroShotCLIP(cfg.clip.arch, cfg.clip.pretrained, cfg.data.class_names,
                                 cfg.clip.templates, torch.device("cpu"))
            info = {"params_total": count_parameters(model.model),
                    "params_inference": count_parameters(model.model.visual)}  # text side precomputed
            size = 224
        else:
            model, ckpt = load_classifier(runs / key / "best.pt")
            info = {"params_total": count_parameters(model), "params_inference": count_parameters(model)}
            size = ckpt["preprocess"].image_size
        for dev in devices:
            print(f"  latency {key} on {dev.type}...")
            lat = torch_latency(model, dev, size, warmup=10, iters=50)
            info[f"{dev.type}_mean_ms"], info[f"{dev.type}_p95_ms"] = lat["mean_ms"], lat["p95_ms"]
        out[key] = info
        del model
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+", default=["cnn", "vit", "clip"])
    parser.add_argument("--runs-dir")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--top-errors", type=int, default=16)
    args = parser.parse_args()

    cfg = load_config(args.config)
    runs = Path(args.runs_dir or cfg.train.runs_dir)
    out = cfg.data.reports_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    names = cfg.data.class_names
    test = get_splits(cfg)["test"]
    images_dir = food101_dir(cfg.data.root) / "images"
    manifest_labels = np.array([s.label for s in test])

    preds: dict[str, Predictions] = {}
    for key in args.models:
        p = Predictions.load(key, runs / key / "test_predictions.npz")
        # Predictions are stored in test-manifest order; make sure that still holds.
        if p.class_names != names or not np.array_equal(p.labels, manifest_labels):
            raise RuntimeError(f"{key}: predictions don't match the current test split/classes. "
                               "Re-run training/zero-shot with the current config.")
        preds[key] = p

    summary_md: list[str] = []
    rows = []
    for key, p in preds.items():
        cm = confusion_matrix(p.labels, p.preds, len(names))
        report = per_class_report(cm, names)
        report.round(4).to_csv(out / f"{key}_per_class.csv")
        plot_confusion(cm, names, f"{DISPLAY.get(key, key)}: test confusion matrix (%)",
                       out / f"{key}_confusion.png")

        pairs = most_confused_pairs(cm, names, k=10)
        pairs.to_csv(out / f"{key}_confused_pairs.csv", index=False)

        worst = worst_errors(p, k=args.top_errors)
        worst.round(4).to_csv(out / f"{key}_worst_errors.csv", index=False)
        image_grid([images_dir / test[i].rel_path for i in worst["index"]],
                   [f"true: {r.true}\npred: {r.pred} ({r.pred_prob:.0%})" for r in worst.itertuples()],
                   out / f"{key}_worst_errors.jpg",
                   title=f"{DISPLAY.get(key, key)}: most confidently wrong")

        metrics = json.loads((runs / key / "test_metrics.json").read_text())
        rows.append({"key": key, "model": DISPLAY.get(key, key), "top1": metrics["top1"],
                     "top5": metrics["top5"], "macro_f1": report.loc["macro avg", "f1"],
                     "errors": int((~p.correct).sum())})

        worst_cls = report.drop(["macro avg", "weighted avg"]).sort_values("f1").head(5)
        summary_md += [f"### {DISPLAY.get(key, key)}", "",
                       "Lowest-F1 classes:", "", worst_cls.round(3).to_markdown(), "",
                       "Most-confused pairs:", "", pairs.head(5).to_markdown(index=False), ""]

    comp = pd.DataFrame(rows).set_index("key")
    if not args.skip_latency:
        print("Benchmarking latency (batch 1, fp32, forward pass only)...")
        comp = comp.join(pd.DataFrame(latency_rows(list(preds), cfg, runs)).T)
    comp.round(4).to_csv(out / "comparison.csv")

    sig = []
    for a, b in combinations(preds, 2):
        r = mcnemar_exact(preds[a].correct, preds[b].correct)
        sig.append({"model_a": a, "model_b": b, "top1_a": preds[a].correct.mean(),
                    "top1_b": preds[b].correct.mean(), **r})
    sig_df = pd.DataFrame(sig)
    sig_df.round(4).to_csv(out / "significance.csv", index=False)

    all_wrong = np.flatnonzero(np.logical_and.reduce([~p.correct for p in preds.values()]))
    aw = pd.DataFrame({"index": all_wrong, "true": [names[manifest_labels[i]] for i in all_wrong],
                       **{f"pred_{k}": [names[p.preds[i]] for i in all_wrong] for k, p in preds.items()}})
    aw.to_csv(out / "all_wrong.csv", index=False)
    show = aw.head(16)
    image_grid([images_dir / test[i].rel_path for i in show["index"]],
               [f"true: {r['true']}\n" + "\n".join(f"{k}: {r[f'pred_{k}']}" for k in preds)
                for _, r in show.iterrows()],
               out / "all_wrong.jpg", title=f"Wrong for every model ({len(aw)} of {len(test)} test images)")

    # Markdown summary
    table = comp.reset_index(drop=True).copy()
    table["top1"] = (100 * table["top1"]).map("{:.2f}%".format)
    table["top5"] = (100 * table["top5"]).map("{:.2f}%".format)
    table["macro_f1"] = table["macro_f1"].map("{:.4f}".format)
    for c in table.columns:
        if c.startswith("params"):
            table[c] = (table[c].astype(float) / 1e6).map("{:.1f}M".format)
        elif c.endswith("_ms"):
            table[c] = table[c].astype(float).map("{:.1f}".format)
    comparison_md = table.to_markdown(index=False)
    (out / "comparison.md").write_text(comparison_md + "\n")
    md = ["# Evaluation summary", "", f"Test split: {len(test)} images, {len(names)} classes.", "",
          "## Comparison", "", comparison_md, "",
          "Latency: batch 1, fp32, model forward pass only (no image decoding/resizing), "
          "mean / p95 over 50 runs after 10 warm-up runs.", "",
          f"Environment: torch {torch.__version__}, {torch.get_num_threads()} CPU threads, "
          f"oneDNN/MKLDNN available: {torch.backends.mkldnn.is_available()}.", "",
          *([] if torch.backends.mkldnn.is_available() else [
              "> Without oneDNN, PyTorch's CPU path runs depthwise convolutions through a slow "
              "generic fallback (`aten::_slow_conv2d_forward`, one call per channel group), which "
              "penalises EfficientNet far more than ViTs or ResNets. CPU numbers for depthwise-conv "
              "models reflect this build, not the architecture's FLOPs.", ""]),
          "## Pairwise significance (exact McNemar)", "", sig_df.round(4).to_markdown(index=False), "",
          f"## Wrong for every model: {len(aw)} images", "", "## Per model", "", *summary_md]
    (out / "summary.md").write_text("\n".join(md))
    print("\n" + comparison_md)
    print(f"\nOutputs in {out}/")


if __name__ == "__main__":
    main()
