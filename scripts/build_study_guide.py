"""Build the personal study guide PDF: docs/food-vision-guide.pdf.

Every number in the guide is read from the recorded files at build time
(configs/, runs/*/test_metrics.json, runs/*/metrics.csv, reports/eda|eval|explain|deploy),
and every code excerpt is extracted from the real source files, so the guide
cannot drift from the repo.

Pipeline: facts + figures -> Markdown -> HTML (+CSS) -> PDF via WeasyPrint.

Requirements (not part of requirements.txt; personal tooling):
    pip install markdown pygments pypdfium2
    WeasyPrint: `brew install weasyprint` (CLI) or `pip install weasyprint`

Usage:
    python scripts/build_study_guide.py [--out docs/food-vision-guide.pdf] [--keep-build DIR]
"""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from datetime import date
from pathlib import Path

import markdown
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402
from pygments.formatters import HtmlFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["cnn", "vit", "clip"]
NAME = {"cnn": "EfficientNet-B0", "vit": "ViT-S/16", "clip": "CLIP ViT-B/32"}
LONG = {"cnn": "EfficientNet-B0 (fine-tuned)", "vit": "ViT-S/16 (fine-tuned)", "clip": "CLIP ViT-B/32 (zero-shot)"}


# =========================================================================== facts


def rd_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def load_facts() -> dict:
    """Everything numeric the guide states, read from the recorded files."""
    F: dict = {}
    raw_cfg = (ROOT / "configs/default.yaml").read_text()
    F["cfg"] = cfg = yaml.safe_load(raw_cfg)
    F["classes"] = sorted(cfg["data"]["classes"])
    # Exclusions with their YAML comments (the reason each image was removed).
    F["exclude"] = [(m.group(1), m.group(2).strip()) for m in
                    re.finditer(r"^\s*-\s*([a-z_]+/\d+)\s*#\s*(.*)$", raw_cfg, re.M)]

    F["eda"] = rd_json("reports/eda/summary.json")
    F["test"] = {m: rd_json(f"runs/{m}/test_metrics.json") for m in MODELS}
    F["curves"] = {m: pd.read_csv(ROOT / f"runs/{m}/metrics.csv") for m in ("cnn", "vit")}
    F["comp"] = pd.read_csv(ROOT / "reports/eval/comparison.csv", index_col="key")
    F["sig"] = pd.read_csv(ROOT / "reports/eval/significance.csv")
    F["per_class"] = {m: pd.read_csv(ROOT / f"reports/eval/{m}_per_class.csv", index_col=0) for m in MODELS}
    F["pairs"] = {m: pd.read_csv(ROOT / f"reports/eval/{m}_confused_pairs.csv") for m in MODELS}
    F["worst"] = {m: pd.read_csv(ROOT / f"reports/eval/{m}_worst_errors.csv") for m in MODELS}
    F["all_wrong"] = len(pd.read_csv(ROOT / "reports/eval/all_wrong.csv"))
    F["artifact"] = rd_json("reports/explain/artifact_check.json")
    F["onnx"] = {m: rd_json(f"reports/deploy/onnx_verify_{m}.json") for m in ("cnn", "vit")}
    bench = rd_json("reports/deploy/benchmark.json")
    F["bench_env"] = bench["environment"]
    F["bench"] = {(r["model"], r["backend"].split(" (")[0]): r for r in bench["results"]}
    F["nutrition"] = json.loads((ROOT / "src/food_vision/resources/nutrition.json").read_text())
    F["yolo"] = rd_json("reports/deploy/yolo_verify.json")
    det_src = ast.parse((ROOT / "src/food_vision/detection.py").read_text())
    F["food_labels"] = next(ast.literal_eval(n.value) for n in det_src.body
                            if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "FOOD_LABELS")

    out = subprocess.run([sys.executable, "-m", "pytest", "-o", "addopts=", "--collect-only", "-q"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests? collected", out)
    per_file = [int(n) for n in re.findall(r"^tests/\S+\.py: (\d+)$", out, re.M)]
    F["n_tests"] = int(m.group(1)) if m else sum(per_file)
    if not F["n_tests"]:
        raise RuntimeError(f"Could not count tests from pytest output:\n{out[-500:]}")
    F["n_det_tests"] = len(re.findall(r"^tests/test_detection\.py::", out, re.M))

    # Derived facts used in the prose.
    pc = F["per_class"]
    F["val_test_gap"] = {
        "cnn": F["test"]["cnn"]["top1"] - F["test"]["cnn"]["best_val_top1"],
        "vit": F["test"]["vit"]["top1"] - F["test"]["vit"]["best_val_top1"],
        "clip": F["test"]["clip"]["top1"] - F["test"]["clip"]["val_top1"],
    }
    F["steak_pair"] = {m: F["pairs"][m].query("class_a=='filet_mignon' and class_b=='steak'").iloc[0]
                       for m in MODELS}
    F["icecream_pair"] = {m: F["pairs"][m].query("class_a=='frozen_yogurt' and class_b=='ice_cream'").iloc[0]
                          for m in MODELS}
    F["recall"] = {m: pc[m]["recall"] for m in MODELS}
    F["final_train_top1"] = {m: F["curves"][m]["train_top1"].iloc[-1] for m in ("cnn", "vit")}
    return F


def pct(x: float, d: int = 2) -> str:
    return f"{100 * x:.{d}f}%"


def bms(F: dict, model: str, backend: str, stat: str = "mean_ms") -> float:
    return F["bench"][(model, backend)][stat]


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    table = "\n".join(out) + "\n"
    if len(rows) <= 12:  # short tables: never split across pages
        return f'\n<div class="keep" markdown="1">\n\n{table}\n</div>\n'
    return table


# =========================================================================== source excerpts


def excerpt(rel: str, qualname: str) -> str:
    """Exact source of a function/method/class from the repo (via the AST)."""
    src = (ROOT / rel).read_text()
    tree = ast.parse(src)
    parts = qualname.split(".")
    nodes = tree.body
    node = None
    for p in parts:
        node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                    and n.name == p)
        nodes = node.body
    start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
    lines = src.splitlines()[start - 1: node.end_lineno]
    return textwrap.dedent("\n".join(lines))


def excerpt_between(rel: str, start: str, end: str) -> str:
    """Exact source lines from the line containing `start` up to (excluding) the line containing `end`."""
    lines = (ROOT / rel).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if start in l)
    j = next(k for k in range(i + 1, len(lines)) if end in lines[k])
    return textwrap.dedent("\n".join(lines[i:j])).rstrip()


def code(src: str, lang: str = "python", title: str = "") -> str:
    head = f'<p class="codetitle">{html.escape(title)}</p>\n\n' if title else ""
    return f"{head}```{lang}\n{src}\n```\n"


# =========================================================================== figures


def find_white_cut(img: Image.Image, near: float) -> int:
    """Row index of the whitest horizontal band closest to `near` (fraction of height)."""
    a = np.asarray(img.convert("L"), dtype=np.float32)
    whiteness = (a > 245).mean(axis=1)
    h = a.shape[0]
    candidates = np.flatnonzero(whiteness > 0.985)
    target = int(near * h)
    return int(candidates[np.argmin(np.abs(candidates - target))]) if len(candidates) else target


def side_by_side(src: Path, dst: Path, title_crop: float = 0.0) -> Path:
    """Split a tall figure at the white band nearest its middle and place the halves side by side."""
    img = Image.open(src).convert("RGB")
    top = 0
    if title_crop:  # find where the title text ends: first white row after the first non-white row
        a = np.asarray(img.convert("L"), dtype=np.float32)
        white = (a > 245).mean(axis=1) > 0.995
        first_ink = int(np.argmax(~white))
        top = first_ink + int(np.argmax(white[first_ink:]))
        if top > title_crop * 4 * img.height:  # safety: never crop more than a thin title strip
            top = 0
    body = img.crop((0, top, img.width, img.height))
    cut = find_white_cut(body, 0.5)
    a, b = body.crop((0, 0, body.width, cut)), body.crop((0, cut, body.width, body.height))
    out = Image.new("RGB", (a.width * 2 + 20, max(a.height, b.height)), "white")
    out.paste(a, (0, 0))
    out.paste(b, (a.width + 20, 0))
    out.save(dst, quality=90)
    return dst


def fig_architecture(F: dict, dst: Path) -> Path:
    """End-to-end pipeline diagram (data -> training -> evaluation -> export -> API)."""
    t, s = F["test"], F["eda"]["images_per_split"]
    stages = [
        ("1  Data", "#E8F0FB", "#4C72B0", [
            "Food-101 (torchvision)", f"{len(F['classes'])}-class subset",
            "seeded, leak-free splits", f"train {s['train']} / val {s['val']}", f"test {s['test']}",
            "EDA: corrupt + duplicate", "checks (MD5)"]),
        ("2  Models", "#FDF0E6", "#DD8452", [
            "EfficientNet-B0", "  fine-tuned, torchvision", "ViT-S/16", "  fine-tuned (timm)",
            "CLIP ViT-B/32", "  zero-shot (OpenCLIP)", "same splits + metrics"]),
        ("3  Evaluation", "#E9F5EC", "#55A868", [
            "top-1/5, per-class F1", "confusion matrices", "McNemar significance",
            "latency (MPS + CPU)", "Grad-CAM (CNN)", "attention rollout (ViT)", "artifact-token check"]),
        ("4  Export", "#F1ECF8", "#8172B3", [
            "ViT-S/16 -> ONNX", "dynamic batch, opset 18", "classes + preprocessing", "  in ONNX metadata",
            f"verified on {F['onnx']['vit']['conversion']['num_images']} test", "  images vs PyTorch",
            "PyTorch vs ORT latency"]),
        ("5  Serving", "#FBEAEA", "#C44E52", [
            "FastAPI  POST /predict", "NumPy/PIL preprocessing", "  (tested = torchvision)",
            "ONNX Runtime (CPU)", "top-3 + nutrition", "Docker image", "  (no PyTorch)"]),
    ]
    fig, ax = plt.subplots(figsize=(12.6, 5.6))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    w, gap, x0, y0, h = 2.22, 0.33, 0.12, 1.15, 4.1
    for i, (title, fill, edge, lines) in enumerate(stages):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fill, ec=edge, lw=1.8))
        ax.text(x + w / 2, y0 + h - 0.32, title, ha="center", va="center", fontsize=14,
                fontweight="bold", color=edge)
        for k, line in enumerate(lines):
            ax.text(x + 0.12, y0 + h - 0.8 - k * 0.45, line, ha="left", va="center", fontsize=10.8,
                    color="#222")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, y0 + h / 2), (x + w + gap - 0.03, y0 + h / 2),
                                         arrowstyle="-|>", mutation_scale=18, lw=1.8, color="#555"))
    # nutrition feeding the API, tests underneath everything
    xa = x0 + 4 * (w + gap)
    ax.add_patch(FancyBboxPatch((xa, 0.22), w, 0.6, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#FFF7D6", ec="#C9A227", lw=1.5))
    ax.text(xa + w / 2, 0.52, "nutrition.json (approx.)", ha="center", va="center", fontsize=10.8)
    ax.add_patch(FancyArrowPatch((xa + w / 2, 0.84), (xa + w / 2, y0 - 0.02), arrowstyle="-|>",
                                 mutation_scale=15, lw=1.5, color="#C9A227"))
    ax.add_patch(FancyBboxPatch((x0, 0.22), 4 * (w + gap) - gap, 0.6,
                                boxstyle="round,pad=0.02,rounding_size=0.1", fc="#F2F2F2", ec="#999", lw=1.2))
    ax.text(x0 + (4 * (w + gap) - gap) / 2, 0.52,
            f"pytest suite ({F['n_tests']} tests): splits, transforms, preprocessing parity, nutrition, API",
            ha="center", va="center", fontsize=10.8, color="#333")
    fig.tight_layout(pad=0.2)
    fig.savefig(dst, dpi=200)
    plt.close(fig)
    return dst


def fig_training_curves(F: dict, dst: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for key, color in [("cnn", "#4C72B0"), ("vit", "#DD8452")]:
        m = F["curves"][key]
        axes[0].plot(m.epoch, m.train_loss, "--", color=color, alpha=0.6, label=f"{NAME[key]} train")
        axes[0].plot(m.epoch, m.val_loss, color=color, label=f"{NAME[key]} val")
        axes[1].plot(m.epoch, m.train_top1, "--", color=color, alpha=0.6, label=f"{NAME[key]} train")
        axes[1].plot(m.epoch, m.val_top1, color=color, label=f"{NAME[key]} val")
        best = m.loc[m.val_top1.idxmax()]
        axes[1].plot(best.epoch, best.val_top1, "*", color=color, ms=14)
    axes[0].set(title="Loss (train loss includes label smoothing)", xlabel="epoch")
    axes[1].set(title="Top-1 accuracy (star = best val epoch = checkpoint used)", xlabel="epoch", ylim=(0.5, 1.0))
    for ax in axes:
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(dst, dpi=170)
    plt.close(fig)
    return dst


def fig_per_class_f1(F: dict, dst: Path) -> Path:
    f1 = pd.DataFrame({m: F["per_class"][m].drop(["macro avg", "weighted avg"])["f1"] for m in MODELS})
    f1 = f1.sort_values("vit")
    ax = f1.plot.barh(figsize=(9, 7.5), width=0.82, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set(xlabel="F1 on the test split", xlim=(0.6, 1.0), title="Per-class F1 (sorted by ViT-S/16)")
    ax.legend([LONG[m] for m in f1.columns], fontsize=8.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(dst, dpi=170)
    plt.close()
    return dst


def fig_detection_demo(F: dict, dst: Path) -> Path | None:
    """Run the real detector + classifier on two test images and draw the results.

    The per-dish numbers are stored in F["demo"] so the prose quotes this run's output.
    Returns None (figure omitted) if the ONNX models aren't exported locally.
    """
    det_path, clf_path = ROOT / "models/yolo11n.onnx", ROOT / "models/vit.onnx"
    if not (det_path.exists() and clf_path.exists()):
        F["demo"] = None
        return None
    from food_vision.detection import OnnxYoloDetector, detect_and_classify, draw_dishes
    from food_vision.inference import OnnxClassifier

    det, clf = OnnxYoloDetector(det_path), OnnxClassifier(clf_path)
    images = ROOT / "data/food-101/images"
    multi = images / "samosa/3661763.jpg"
    # First filet mignon image in the official test list with no food region -> whole-image fallback.
    test_list = json.loads((ROOT / "data/food-101/meta/test.json").read_text())["filet_mignon"]
    fallback = None
    for rel in test_list[:50]:
        with Image.open(images / f"{rel}.jpg") as im:
            if not det.detect(im):
                fallback = images / f"{rel}.jpg"
                break
    panels, demo = [], {}
    for key, path in (("multi", multi), ("fallback", fallback)):
        if path is None:
            continue
        with Image.open(path) as im:
            dishes = detect_and_classify(im, det, clf, top_k=1)
            panels.append(draw_dishes(im, dishes))
        demo[key] = {"path": str(path.relative_to(images)), "dishes": [
            (d.detection.label if d.detection else None, d.detection.score if d.detection else None,
             d.predictions[0][0], d.predictions[0][1]) for d in dishes]}
    F["demo"] = demo
    h = 420
    panels = [p.resize((round(p.width * h / p.height), h), Image.BILINEAR) for p in panels]
    canvas = Image.new("RGB", (sum(p.width for p in panels) + 12 * (len(panels) - 1), h), "white")
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + 12
    canvas.save(dst, quality=92)
    return dst


def make_figures(F: dict, build: Path) -> dict[str, Path]:
    r = ROOT / "reports"
    return {
        "detect": fig_detection_demo(F, build / "detect_demo.jpg"),
        "arch": fig_architecture(F, build / "architecture.png"),
        "curves": fig_training_curves(F, build / "curves.png"),
        "f1": fig_per_class_f1(F, build / "f1.png"),
        "grid": side_by_side(r / "eda/sample_grid.jpg", build / "sample_grid_2col.jpg"),
        "correct": side_by_side(r / "explain/correct.jpg", build / "correct_2col.jpg", title_crop=0.012),
        "errors": side_by_side(r / "explain/errors.jpg", build / "errors_2col.jpg", title_crop=0.009),
        "counts": r / "eda/class_counts.png",
        "sizes": r / "eda/image_sizes.png",
        "cm_vit": r / "eval/vit_confusion.png",
        "cm_clip": r / "eval/clip_confusion.png",
        "worst_vit": r / "eval/vit_worst_errors.jpg",
        "all_wrong": r / "eval/all_wrong.jpg",
        "dup": ROOT / "data/food-101/images/omelette/2619753.jpg",
    }


def fig(path: Path, caption: str, width: int = 100, cls: str = "") -> str:
    if not Path(path).exists():
        return f'<p class="missing">[figure not found: {html.escape(str(path))}]</p>\n'
    return (f'\n<figure class="{cls}"><img src="{Path(path).resolve().as_uri()}" style="width:{width}%">'
            f"<figcaption>{caption}</figcaption></figure>\n\n")


def box(kind: str, title: str, body_md: str) -> str:
    """A shaded callout (concept / why / interview tip / caveat) with Markdown inside."""
    return f'\n<div class="box {kind}" markdown="1">\n<p class="boxtitle">{title}</p>\n\n{body_md}\n\n</div>\n\n'


# =========================================================================== content


def sec_overview(F: dict) -> str:
    t, c = F["test"], F["comp"]
    rows = []
    for m in MODELS:
        train = (f"fine-tuned, best epoch {t[m]['best_epoch']} of {t[m]['epochs_trained']}" if m != "clip"
                 else "none (zero-shot)")
        rows.append([LONG[m], train, f"{c.loc[m, 'params_inference'] / 1e6:.1f}M", pct(t[m]["top1"]),
                     pct(t[m]["top5"]), f"{c.loc[m, 'macro_f1']:.3f}",
                     f"{c.loc[m, 'mps_mean_ms']:.1f} ms"])
    sig_vc = F["sig"].query("model_a=='vit' and model_b=='clip'").iloc[0]
    return f"""
# 1. Overview {{#overview}}

**The problem.** Given a photo of a meal, recognise the dish and return useful nutritional context. That makes food logging quick instead of tedious. This repo solves the recognition and nutrition part for **{len(F['classes'])} dishes** from the public Food-101 dataset, including Indian-relevant ones (samosa, chicken curry) alongside common dishes.

**Link to your paper.** The project rebuilds your final-year project and published paper, *"An Integrated Framework for Food Recognition, Nutritional Insights, and Meal Recording"*. The original combined recognition, nutrition lookup and meal recording. This rebuild re-implements recognition and nutrition as an engineering-grade pipeline; meal recording is on the v2 roadmap. The full citation (authors, venue, year, DOI) is still a placeholder in the README.

**What you built**

* a reproducible data pipeline with leak-free splits and data-quality checks;
* three model families trained and evaluated on **identical** splits: a fine-tuned CNN, a fine-tuned Vision Transformer and zero-shot CLIP;
* a full evaluation with significance tests, error analysis and explainability;
* a verified ONNX export served by a FastAPI service in Docker;
* multi-food detection: a YOLO detector (ONNX) proposes food regions and the classifier labels each one;
* a {F['n_tests']}-test pytest suite.

**Headline results** (held-out test split, {F['eda']['images_per_split']['test']} images, 100 per class)

{md_table(["Model", "Training", "Params*", "Top-1", "Top-5", "Macro-F1", "Latency (MPS)"], rows)}

<p class="small">*Parameters used at inference (CLIP: image encoder only; its text side is precomputed). Latency: batch 1, fp32, forward pass only, mean of 50 runs.</p>

**Key decisions in one line each**

* **Food-101 subset ({len(F['classes'])} classes, {F['cfg']['data']['train_per_class']} train images per class):** fast to train on a laptop, while keeping deliberately confusable pairs.
* **Train/val drawn from the official train list, test from the official test list:** no leakage, and results stay comparable to the standard protocol.
* **Three model families on the same splits:** a fair answer to "CNN vs ViT vs foundation model?".
* **Best-val checkpoint evaluated once on test:** the test set never influences training decisions.
* **Exact McNemar tests:** they show that CLIP ({pct(t['clip']['top1'])}) and the ViT ({pct(t['vit']['top1'])}) are statistically tied (p = {sig_vc['p_value']:.2f}).
* **ViT-S/16 deployed:** tied with CLIP on accuracy, with 4× fewer inference parameters and lower latency.
* **ONNX + ONNX Runtime + FastAPI + Docker:** a portable model file and a small serving image with no PyTorch, with preprocessing proven identical to training.
* **Detect-then-classify for multi-dish photos:** Food-101 has no boxes, so a COCO-pretrained YOLO11n proposes food regions and the verified ViT labels each crop. The NumPy re-implementation is verified against Ultralytics.
* **Numbers only from recorded files:** every figure in this guide is read from `runs/` and `reports/` at build time.
"""


def sec_architecture(F: dict, figs: dict) -> str:
    return f"""
# 2. Architecture {{#architecture}}

The project is a straight pipeline. Each stage is a script in `scripts/` that reads the previous stage's outputs from disk, and all the logic lives in the `food_vision` package under `src/`.

{fig(figs['arch'], "End-to-end pipeline. Every stage writes files the next stage reads (data/splits → runs/ → reports/ → models/*.onnx → API).", 100)}

{md_table(["Stage", "Script", "Reads", "Writes"], [
    ["Data", "`prepare_data.py`, `eda.py`", "Food-101 download, config", "`data/splits/*.csv`, `reports/eda/`"],
    ["Models", "`train.py --model cnn` / `vit`, `zero_shot_clip.py`", "split manifests", "`runs/<model>/` (metrics, checkpoint, test logits)"],
    ["Evaluation", "`evaluate.py`, `explain.py`", "`runs/*/test_predictions.npz`, checkpoints", "`reports/eval/`, `reports/explain/`"],
    ["Export", "`export_onnx.py`, `benchmark.py`", "`runs/<model>/best.pt`", "`models/<model>.onnx`, `reports/deploy/`"],
    ["Detection", "`export_yolo.py`, `detect.py`", "Ultralytics `yolo11n.pt`, test images", "`models/yolo11n.onnx`, `reports/deploy/yolo_verify.json`"],
    ["Serving", "`uvicorn food_vision.api.app:app`, Dockerfile", "`models/vit.onnx`, `models/yolo11n.onnx` (optional), `nutrition.json`", "JSON responses (`/predict`, `/detect`)"],
])}

**One request through the API**

1. `POST /predict` receives the image upload, then checks its content type and size.
2. Pillow decodes it. Unreadable images are rejected with HTTP 400.
3. **Preprocessing in NumPy:** resize the shorter side to 256 px, center-crop 224×224, scale to [0, 1], normalise. The result is identical to the torchvision transform used in training, and a unit test enforces this.
4. **ONNX Runtime** runs the ViT and returns 20 logits. Softmax turns them into probabilities, and the top-k are kept (k = 3 by default).
5. Each predicted dish is looked up in `nutrition.json`, and the JSON response includes the approximate-values disclaimer.

`POST /detect` runs the same steps once per food region found by the YOLO detector (section 3.6).

{box("why", "Why a pipeline of scripts, not one big notebook?", "Each step can be rerun on its own, its outputs are inspectable files, and the package code can be imported by tests and by the API. Notebooks in this repo only *display* results; they never contain logic.")}
"""


def sec_phase1(F: dict, figs: dict) -> str:
    d, e = F["cfg"]["data"], F["eda"]
    excl = "\n".join(f"* `{img}`: {why}" for img, why in F["exclude"])
    return f"""
# 3. Phase-by-phase walkthrough {{#phases}}

## 3.1 Phase 1: Data and EDA {{#phase1}}

**What was done.**

* Downloaded Food-101.
* Selected a configurable subset of {len(F['classes'])} classes.
* Sampled train, val and test images deterministically, and saved the split as CSV manifests.
* Ran exploratory data analysis (EDA): class counts, image sizes, a sample grid, decode checks and duplicate detection.

{box("concept", "Concept: the dataset and its official split", f"""
**Food-101** has 101 dishes with 1,000 images each, split officially into **750 train / 250 test per class**. Its authors left the training images deliberately uncleaned (some wrong labels, intense colours) but manually cleaned the test images. This detail explains a result later in the guide.

The images are web photos rescaled so the longest side is at most 512 px. The EDA confirms this: the longest side is exactly 512 px for {e['pct_longest_side_eq_512']:.0f}% of the {sum(e['images_per_split'].values())} images used.""")}

**The subset.** The {len(F['classes'])} classes, listed in `configs/default.yaml`, are:
{", ".join(F['classes'])}.

The list includes Indian-relevant dishes, plus three **deliberately confusable pairs** (steak / filet mignon, ice cream / frozen yogurt, chocolate cake vs other desserts), so the error analysis has something meaningful to explain.

Per class: **{d['train_per_class']} train, {d['val_per_class']} val, {d['test_per_class']} test**, giving **{e['images_per_split']['train']} / {e['images_per_split']['val']} / {e['images_per_split']['test']}** images in total.

{box("concept", "Concept: train, validation and test splits, and leakage", """
* **Train** is what the model learns from.
* **Validation (val)** is what *you* use to make decisions: when to stop, which checkpoint to keep, which hyperparameters to use.
* **Test** is touched once, at the end, to estimate real-world accuracy.

**Leakage** means information from val or test reaching training, for example the same photo in both train and test. It inflates scores and is the most common silent bug in ML projects.""")}

**How the split is built** (`build_splits` in `src/food_vision/data.py`)

* **val + train** are sampled *without overlap* from the official **train** list, and **test** from the official **test** list. So no test image can ever be trained on, and results are comparable to the standard Food-101 protocol.
* **Deterministic:** each class gets its own random generator seeded with `"{{seed}}-{{class}}"`. The same config always gives the same split, and adding a class does not reshuffle the others.
* **Labels** are assigned in **sorted** class order, so reordering the YAML list doesn't change the label ids.
* **Fingerprint:** the split is saved as CSV manifests (`data/splits/train.csv`, …) with a hash of every setting that affects it (seed, classes, counts, exclusions). If the config changes, the fingerprint no longer matches and the split is rebuilt automatically.

**Robust download.** The first download attempt failed when the ETH server reset the connection partway through, and torchvision's downloader restarted from zero. `download_resumable` replaced it:

* it uses HTTP **Range** requests to continue from the bytes already on disk;
* it retries with exponential backoff;
* it verifies the MD5 checksum and extracts with tarfile's safe `data` filter;
* it writes an `.extracted` marker only after extraction completes.

**EDA findings** (`scripts/eda.py` → `reports/eda/`)

* **{e['corrupt_images']} corrupt images.** Every file was fully decoded, not just header-checked.
* **Non-RGB images:** {'none' if not e['non_rgb_images'] else e['non_rgb_images']} in this subset. The Dataset converts everything to RGB anyway.
* **Sizes:** width {e['width']['min']:.0f}–{e['width']['max']:.0f} px, height {e['height']['min']:.0f}–{e['height']['max']:.0f} px. Only {e['pct_shorter_side_lt_224']}% of images have a shorter side below 224 px, so upsampling is rare.
* **Duplicates:** every file was hashed (MD5). The check found **two real problems**, now excluded via the config's `exclude` list:

{excl}

* After exclusion: **{e['duplicate_groups']} duplicate groups, {e['cross_split_duplicate_groups']} across splits.**

{box("why", "Why the duplicate check matters", "The omelette/pancakes file is one photo with two labels; it actually shows both dishes, so no single-label classifier can be right on it. The dumplings duplicate was leakage between train and val, which would make validation look slightly better than reality. Both are the kind of data bug interviewers love to hear you caught yourself.")}

{fig(figs['dup'], "The duplicate filed as both omelette/2619753 and pancakes/2619752: one photo showing both dishes.", 42)}

{fig(figs['grid'], "Random training images per class (reports/eda/sample_grid.jpg, split into two columns). Note the variety in plating, lighting and framing, and the occasional odd label.", 100)}

{fig(figs['sizes'], "Image dimensions (reports/eda/image_sizes.png). The longest side is always 512 px, and most images are square, 4:3 or 3:4.", 100)}

{fig(figs['counts'], "Images per class per split (reports/eda/class_counts.png): perfectly balanced by construction.", 78)}

**Transforms** (`build_transforms` in `data.py`)

* **Training:** `RandomResizedCrop(224, scale=(0.5, 1.0))`, `RandomHorizontalFlip`, `ColorJitter(0.2, 0.2, 0.2, 0.02)`, `ToTensor`, `Normalize`.
* **Evaluation:** `Resize(256)` → `CenterCrop(224)` → `ToTensor` → `Normalize`. The ratio 224/256 = 0.875 is the standard ImageNet evaluation recipe.

{box("concept", "Concept: data augmentation", """
Augmentation shows the model a slightly different version of each image every epoch. That teaches it which changes do **not** affect the label:

* **RandomResizedCrop:** zoom and framing (a dish is still a dish if you crop half the plate).
* **Horizontal flip:** mirror images are still the same food.
* **ColorJitter:** lighting and white balance differ between phones and restaurants.

Augmentation only happens at training time. Evaluation is deterministic, so results are reproducible.""")}

{box("concept", "Concept: normalisation", "Pixel values are scaled to [0, 1], then each colour channel has the pretraining dataset's mean subtracted and is divided by its standard deviation. Pretrained weights expect inputs with those statistics. This repo uses the ImageNet mean/std for EfficientNet and 0.5/0.5 for the timm ViT, each model's own values, which are stored inside its checkpoint and ONNX file.")}
"""


def sec_phase2(F: dict, figs: dict) -> str:
    t, cfg = F["test"], F["cfg"]
    tr, mc = cfg["train"], cfg["models"]
    curves = F["curves"]
    return f"""
## 3.2 Phase 2: Three models, same data, same evaluation {{#phase2}}

**What was done.** Two pretrained image classifiers were fine-tuned on the training split, and CLIP was evaluated zero-shot. All three were evaluated on the same test images with the same metric code.

{box("concept", "Concept: transfer learning and fine-tuning", f"""
Training a vision model from scratch needs millions of images. **Transfer learning** starts from a network already trained on a huge dataset (ImageNet). Its early layers already detect edges, textures and shapes, and its later layers detect object parts. You replace the final classification layer (the **head**) with a new one sized for your classes (here 20) and continue training (**fine-tuning**) on your small dataset.

With {cfg['data']['train_per_class']} images per class, this is the only sensible approach. In this repo, all layers are fine-tuned (none are frozen).""")}

### Model 1: EfficientNet-B0 (CNN)

{box("concept", "Concept: convolutional neural networks (CNNs)", """
A **convolution** slides a small learned filter (for example 3×3) across the image and responds wherever its pattern appears. Stacking many convolution layers, with downsampling in between, builds a hierarchy: edges, then textures, then parts, then objects.

* **Built-in assumptions:** CNNs have strong **inductive biases**. They assume *locality* (nearby pixels matter together) and *translation equivariance* (a pattern means the same thing anywhere in the image). These assumptions make CNNs data-efficient.
* **EfficientNet-B0:** uses **depthwise-separable convolutions**, which filter each channel separately and then mix channels with 1×1 convolutions. This is far cheaper than standard convolutions, and it becomes relevant again in the benchmark section.""")}

* **Model:** `torchvision.models.efficientnet_b0` with ImageNet weights. The 1000-class head is replaced by a 20-class `Linear` layer.
* **Size:** {t['cnn']['params'] / 1e6:.1f}M parameters with the 20-class head. The usual 5.3M figure includes the 1000-class head.
* **Learning rate:** {mc['cnn']['lr']}, weight decay {mc['cnn']['weight_decay']}.

### Model 2: ViT-S/16 (Vision Transformer)

{box("concept", "Concept: Vision Transformers, patches and attention", """
A **Vision Transformer** treats an image like a sentence:

* **Patches:** the 224×224 image is cut into 16×16-pixel **patches**, giving 14 × 14 = 196 patches. Each patch is flattened and linearly projected into a vector (a **token**), then a learned **position embedding** is added.
* **CLS token:** an extra learnable **[CLS] token** is prepended. Its final state is used for classification.
* **Self-attention:** each token computes how relevant every other token is to it (query · key similarity, then softmax), and mixes in their values accordingly. So from the very first layer, any patch can use information from any other patch.
* **Trade-off:** ViTs have weaker built-in assumptions than CNNs, so they need more pretraining data. With good pretraining they generalise very well.

You know transformers from LLMs. This is the same architecture; only the tokenizer differs (image patches instead of word pieces).""")}

* **Model:** timm `{mc['vit']['arch']}`. That's ViT-Small, 16×16 patches, pretrained on ImageNet-21k with the AugReg recipe, then fine-tuned on ImageNet-1k.
* **Size:** {t['vit']['params'] / 1e6:.1f}M parameters.
* **Learning rate:** {mc['vit']['lr']}, which is 10× lower than the CNN's {mc['cnn']['lr']}. Transformers are more sensitive to large updates of their pretrained weights. Weight decay {mc['vit']['weight_decay']}.
* **Normalisation:** the model's own preprocessing (mean/std 0.5), read via `timm.data.resolve_model_data_config`.

### Model 3: CLIP ViT-B/32 (zero-shot, no training)

{box("concept", "Concept: CLIP, contrastive learning and zero-shot classification", f"""
**CLIP** has an **image encoder** and a **text encoder**, trained together on hundreds of millions of image–caption pairs (here OpenCLIP's LAION-2B weights, `{mc['clip']['pretrained']}`).

* **Contrastive learning:** in each batch, the model pulls each image's embedding towards *its own* caption's embedding and pushes it away from all other captions. The result is a shared embedding space where an image and a sentence describing it are close.
* **Zero-shot classification:** turn each class name into a sentence (a **prompt**), embed it with the text encoder, and predict the class whose text embedding is most similar (cosine similarity) to the image embedding. No training images are needed.
* **Prompt ensembling:** average the embeddings of several prompt templates per class, which makes the class embedding more robust. This repo uses {len(mc['clip']['templates'])} templates, for example *"{mc['clip']['templates'][0]}"*.

It is the vision equivalent of prompting an LLM instead of fine-tuning it.""")}

* **Implementation:** `ZeroShotCLIP` in `src/food_vision/clip_zero_shot.py`.
* **Class embeddings:** built once at start-up and stored as a matrix. Classification is then one image encoding plus one matrix multiply, scaled by CLIP's learned temperature (`logit_scale`).
* **Same interface:** it outputs logits just like a classifier, so the *same* evaluation code handles all three models.
* **Preprocessing:** CLIP uses its own (bicubic resize, CLIP mean/std).
* **Size:** {F['comp'].loc['clip', 'params_total'] / 1e6:.1f}M parameters in total, of which {F['comp'].loc['clip', 'params_inference'] / 1e6:.1f}M are the image encoder used per photo.

### The training recipe (shared by the CNN and the ViT)

{md_table(["Setting", "Value", "Where"], [
    ["Optimiser", f"AdamW, no weight decay on biases and norm layers", "`engine.build_optimizer`"],
    ["LR schedule", f"linear warmup {tr['warmup_epochs']} epoch, then cosine decay to 1% (per step)", "`engine.warmup_cosine`"],
    ["Loss", f"cross-entropy with label smoothing {tr['label_smoothing']}", "`scripts/train.py`"],
    ["Gradient clipping", f"max norm {tr['grad_clip']}", "`engine.train_one_epoch`"],
    ["Mixed precision", "fp16 autocast + GradScaler on MPS/CUDA (bf16 on CUDA if supported)", "`engine.amp_setup`"],
    ["Epochs / early stopping", f"max {tr['epochs']}, stop after {tr['patience']} epochs without val top-1 gain", "`engine.EarlyStopping`"],
    ["Batch size", f"{cfg['data']['batch_size']}", "`configs/default.yaml`"],
    ["Checkpoint", "best val top-1; then evaluated once on test", "`models.save_checkpoint`"],
])}

{box("concept", "Concept: AdamW and weight decay", """
**Adam** adapts each parameter's step size using running averages of its gradients and squared gradients, so it is robust to badly scaled problems. **Weight decay** pulls weights towards zero to reduce overfitting.

**AdamW** applies decay *directly to the weights* instead of mixing it into the gradient (the original Adam formulation). This matters because Adam's adaptive scaling would otherwise weaken the decay unevenly.

Biases and normalisation parameters are excluded from decay, which is standard practice: shrinking them does not regularise anything useful.""")}

{box("concept", "Concept: learning-rate warmup + cosine schedule", """
* **Warmup:** at the start, the new random head produces large, noisy gradients that can wreck the pretrained weights. So the learning rate ramps up linearly from almost zero over the first epoch.
* **Cosine decay:** after warmup, the learning rate follows half a cosine curve down to 1% of its peak: big steps early to make progress, tiny steps late to settle into a good minimum.

The schedule is updated **every step**, not every epoch, which gives a smooth curve. A unit test checks the exact values.""")}

{box("concept", "Concept: label smoothing", f"Instead of a target of 100% on the true class, the target becomes about {100 * (1 - tr['label_smoothing']):.0f}% on the true class with the rest spread evenly over the others (smoothing {tr['label_smoothing']}). This discourages over-confident predictions and acts as a regulariser, which is useful when some training labels are wrong, as in Food-101. It is why the reported *train* loss can never reach zero. Validation loss is computed without smoothing.")}

{box("concept", "Concept: mixed precision (AMP)", """
Most of the arithmetic runs in 16-bit floats (fp16), which is faster and uses half the memory, while sensitive operations stay in 32-bit floats. This is called **autocast**.

* **The risk:** fp16 has a small range, so tiny gradients underflow to zero.
* **The fix:** a **GradScaler** multiplies the loss by a large factor (starting at 65,536) before the backward pass, then divides the gradients back down. If any gradient overflows to infinity, that step is skipped and the scale is reduced.
* **Seen in practice:** this happened on the ViT's first step and produced a harmless PyTorch warning, which the code now filters with an explanatory comment.
* **bf16 alternative:** bf16 has fp32's range, so it needs no scaler. It is used on CUDA GPUs that support it.""")}

{box("concept", "Concept: early stopping and checkpoint selection", f"After every epoch, the model is evaluated on val. If val top-1 improves, the weights are saved (`best.pt`). If it fails to improve for {tr['patience']} consecutive epochs, training stops. The checkpoint that goes to test is the best-*val* one, never the last epoch, and the test set is used exactly once.")}

**What happened during training** (`runs/*/metrics.csv`)

* **EfficientNet-B0:** ran all {t['cnn']['epochs_trained']} epochs ({t['cnn']['train_minutes']} min). Best val top-1 was {pct(t['cnn']['best_val_top1'])} at epoch {t['cnn']['best_epoch']}.
* **ViT-S/16:** stopped early after {t['vit']['epochs_trained']} epochs ({t['vit']['train_minutes']} min). Best val top-1 was {pct(t['vit']['best_val_top1'])} at epoch {t['vit']['best_epoch']}.
* **Both overfit:** final train top-1 was {pct(F['final_train_top1']['cnn'], 1)} for the CNN and {pct(F['final_train_top1']['vit'], 1)} for the ViT, while val levelled off in the high 80s. That's a big train/val gap, despite augmentation and label smoothing. The train numbers are measured *with* augmentation, so the true gap is even larger.

{fig(figs['curves'], "Training curves from runs/cnn/metrics.csv and runs/vit/metrics.csv. The ViT learns faster and peaks at epoch 5; the CNN improves until epoch 12.", 100)}

**Test results** (written by `metrics.save_test_outputs`)

{md_table(["Model", "Test top-1", "Test top-5", "Val top-1"], [
    [LONG['cnn'], pct(t['cnn']['top1']), pct(t['cnn']['top5']), pct(t['cnn']['best_val_top1'])],
    [LONG['vit'], pct(t['vit']['top1']), pct(t['vit']['top5']), pct(t['vit']['best_val_top1'])],
    [LONG['clip'], pct(t['clip']['top1']), pct(t['clip']['top5']), pct(t['clip']['val_top1'])],
])}

Each run also writes `test_predictions.npz`: the raw logits and labels of every test image, in manifest order. Phase 3 computes **all** metrics from these files, which guarantees identical evaluation for the three models.
"""


def sec_phase3(F: dict, figs: dict) -> str:
    c, sig = F["comp"], F["sig"]
    sp, ic = F["steak_pair"], F["icecream_pair"]
    art = F["artifact"]
    art_rows = [[a["layer"], f"{a['median_max_over_median_norm']:.1f}×", pct(a["mean_topk_overlap"], 0),
                 pct(a["chance_topk_overlap"], 1), f"{a['mean_corr_rollout_vs_norm']:+.2f}"] for a in art]
    sig_rows = [[f"{NAME[r.model_a]} vs {NAME[r.model_b]}", r.only_a_correct, r.only_b_correct, f"{r.p_value:.4f}"]
                for r in sig.itertuples()]
    lat_rows = [[LONG[m], f"{c.loc[m, 'mps_mean_ms']:.1f} / {c.loc[m, 'mps_p95_ms']:.1f}",
                 f"{c.loc[m, 'cpu_mean_ms']:.1f} / {c.loc[m, 'cpu_p95_ms']:.1f}"] for m in MODELS]
    pair_rows = []
    for m in MODELS:
        for r in F["pairs"][m].head(3).itertuples():
            pair_rows.append([NAME[m], f"{r.class_a} ↔ {r.class_b}", r.a_as_b, r.b_as_a, r.total])
    vit_rec = F["recall"]["vit"]
    return f"""
## 3.3 Phase 3: Evaluation and error analysis {{#phase3}}

**What was done.** Detailed metrics, a comparison table with latency, significance tests, error analysis, and visual explanations (Grad-CAM for the CNN, attention rollout for the ViT), plus a check of whether those explanations can be trusted.

{box("concept", "Concept: classification metrics", """
* **Top-1 accuracy:** the share of images whose highest-scoring class is correct. **Top-5:** the true class is among the five highest.
* **Precision for class X:** of everything predicted as X, how much really was X. **Recall for X:** of all real X images, how many were found.
* **F1:** the harmonic mean of precision and recall. It is only high when both are. **Macro-F1** averages F1 over classes equally.
* **Confusion matrix:** row = true class, column = predicted class. The diagonal is correct predictions; off-diagonal cells show *which* classes get mixed up. Rows are normalised to percentages here.""")}

**Comparison and latency** (`reports/eval/comparison.csv`; batch 1, fp32, forward pass only; mean / p95 in ms over 50 runs after 10 warm-up runs)

{md_table(["Model", "Macro-F1", "Errors (of 2000)", "MPS mean / p95", "CPU mean / p95"],
          [[LONG[m], f"{c.loc[m, 'macro_f1']:.4f}", int(c.loc[m, 'errors']),
            *lat_row[1:]] for m, lat_row in zip(MODELS, lat_rows)])}

{box("concept", "Concept: measuring latency properly", "GPUs, including Apple's MPS, run work *asynchronously*: a Python call returns before the GPU finishes. So `benchmark.synchronize()` waits for the device before stopping the clock. Warm-up runs are discarded (the first calls pay for memory allocation and kernel compilation), and both the **mean** and the **p95** are reported. p95 means 95% of requests were at least this fast, and it is what users notice as tail latency.")}

{box("caveat", "The CPU number that looked wrong", f"EfficientNet-B0 took **{c.loc['cnn', 'cpu_mean_ms']:.1f} ms** on CPU, slower than the ViT despite having about 5× fewer parameters. The PyTorch profiler showed the cause: this PyTorch build has **no oneDNN** (recorded in `reports/eval/summary.md`), so depthwise convolutions fall back to a slow generic kernel (`aten::_slow_conv2d_forward`), one call per channel group. The number is real for this build but not representative of the architecture; Phase 4 confirms this with ONNX Runtime.")}

**Is CLIP really better? The McNemar test** (`reports/eval/significance.csv`)

{box("concept", "Concept: McNemar's test", """
Two models were evaluated on the *same* 2,000 images, so their results are paired. Images where both are right, or both are wrong, say nothing about which model is better. Only the **discordant** images count: A right and B wrong, or the reverse.

If the models were truly equal, each discordant image would be a fair coin flip. The **exact McNemar test** asks how surprising the observed split is under that coin-flip assumption (a binomial test). A small p-value (conventionally below 0.05) means the difference is unlikely to be chance.

This is the right test for comparing classifiers on one test set; comparing two accuracy percentages by eye is not.""")}

{md_table(["Comparison", "Only A correct", "Only B correct", "p-value"], sig_rows)}

* **ViT vs CLIP:** p = {sig.query("model_a=='vit' and model_b=='clip'").p_value.iloc[0]:.2f}, **no significant difference.** CLIP's {pct(F['test']['clip']['top1'] - F['test']['vit']['top1'])} lead is within noise.
* **Both beat the CNN:** p = {sig.query("model_a=='cnn' and model_b=='vit'").p_value.iloc[0]:.3f} and {sig.query("model_a=='cnn' and model_b=='clip'").p_value.iloc[0]:.4f}.

**Per-class performance**

{fig(figs['f1'], "Per-class F1 for all three models (from reports/eval/*_per_class.csv).", 82)}

**Most-confused class pairs** (errors in both directions; `reports/eval/*_confused_pairs.csv`)

{md_table(["Model", "Pair (A ↔ B)", "A predicted as B", "B predicted as A", "Total"], pair_rows)}

{fig(figs['cm_vit'], "ViT-S/16 row-normalised confusion matrix (reports/eval/vit_confusion.png). The off-diagonal blocks at steak ↔ filet mignon and ice cream → frozen yogurt stand out.", 88)}

* **steak ↔ filet mignon** is the top confusion for **all three** models ({sp['cnn'].total}, {sp['vit'].total} and {sp['clip'].total} errors). ViT recall is {pct(vit_rec['steak'], 0)} on steak and {pct(vit_rec['filet_mignon'], 0)} on filet mignon.
* **ice cream → frozen yogurt** is much worse for zero-shot CLIP: {ic['clip'].b_as_a} ice-cream images called frozen yogurt, against {ic['vit'].b_as_a} for the fine-tuned ViT. Text prompts can't separate two visually near-identical desserts; labelled examples can.

{fig(figs['cm_clip'], "CLIP zero-shot confusion matrix (reports/eval/clip_confusion.png): note the large ice_cream → frozen_yogurt cell.", 80)}

**The most confidently wrong predictions** are wrong predictions sorted by the probability the model gave its (wrong) answer. These are the most dangerous errors in production, because a confidence threshold won't catch them.

{fig(figs['worst_vit'], "ViT-S/16's most confidently wrong test images (reports/eval/vit_worst_errors.jpg).", 92)}

Many are **multi-dish photos**: ice cream served *on* waffles (labelled ice cream, predicted waffles at 97%), a pastry beside a salad. In these the label is ambiguous rather than the model being clearly wrong, which is the main argument for v2's multi-dish detection.

**{F['all_wrong']} test images are wrong for every model** (`reports/eval/all_wrong.jpg`); several look like label noise.

{fig(figs['all_wrong'], f"Images every model gets wrong ({F['all_wrong']} in total; first 16 shown).", 92)}

### What do the models look at?

{box("concept", "Concept: Grad-CAM", """
**Grad-CAM** asks which regions of the last convolutional feature map pushed the score of the predicted class up.

1. Run the image forward and keep the activations of the last convolutional block. For EfficientNet-B0 that is 1280 channels × 7 × 7.
2. Backpropagate the predicted class's score to get the gradient for each activation.
3. Average each channel's gradients over space. That gives each channel's importance.
4. Take the importance-weighted sum of the channels, apply ReLU (keep only positive evidence), then upsample to 224×224 and overlay it on the image.

`GradCAM` in `src/food_vision/explain.py` implements this by hand with forward and backward hooks.""")}

{box("concept", "Concept: attention rollout", """
A ViT has no convolutional feature map, but it does have attention weights in every layer.

* **How rollout works:** for each layer, average the attention matrix over its heads, add the identity matrix (to account for the residual connection that carries each token forward), re-normalise the rows, and **multiply the matrices across all 12 layers**. The [CLS] row of the result estimates how much each patch contributed to the final [CLS] token.
* **Capturing the weights:** PyTorch's fused attention kernel never materialises the attention matrix, so `AttentionRollout` turns fused attention off and hooks the attention weights after the softmax.
* **Caveat:** rollout is *class-agnostic*. It shows where information flows, not evidence for one particular class.""")}

{fig(figs['correct'], "Correct predictions. For each image: original | CNN Grad-CAM | ViT attention rollout (reports/explain/correct.jpg).", 100)}

Grad-CAM consistently highlights the food: the samosa filling, the curry bowl, the dumpling pleats. Rollout often lights up **background patches** (corners, borders, table). Is that a bug, or the model?

{box("concept", "Concept: artifact tokens and register tokens", """
Darcet et al. (2023, *Vision Transformers Need Registers*) found that large, well-trained ViTs repurpose a few **low-information background patches** as global scratch space.

* **What gives them away:** these tokens have unusually **high norms** and attract a lot of attention, so attention maps highlight them even though they carry no information about the object.
* **The fix:** add a few extra learnable **register tokens** to the input sequence, giving the model dedicated scratch space.""")}

**The test** (`explain.artifact_token_check`, run over 100 test images): measure each patch token's norm in the residual stream *before* the final LayerNorm, and check whether rollout's top-5 patches are the high-norm ones.

{md_table(["After block", "Max / median token norm", "Rollout top-5 ∩ norm top-5", "Expected by chance", "Correlation"], art_rows)}

* **Result:** rollout's hottest patches coincide with the highest-norm tokens about **{max(a['mean_topk_overlap'] for a in art) / art[0]['chance_topk_overlap']:.0f}× more often than chance**, and the outlier tokens reach {max(a['median_max_over_median_norm'] for a in art):.1f}× the median norm. The background hot spots are **artifact tokens**, so plain rollout is misleading on this ViT.
* **Debugging lesson:** the first version of this test measured norms *after* the final LayerNorm, which flattens all token norms and wrongly suggested there were no artifacts. Measuring in the residual stream fixed it.

{fig(figs['errors'], "Explanations on errors (reports/explain/errors.jpg): the most confidently wrong CNN and ViT predictions plus images missed by every model.", 100)}
"""


def sec_phase4(F: dict, figs: dict) -> str:
    ov, oc = F["onnx"]["vit"], F["onnx"]["cnn"]
    env = F["bench_env"]
    backends = ["PyTorch CPU", "ONNX Runtime CPU", "ONNX Runtime CoreML", "PyTorch MPS"]
    rows = [[b, *[f"{bms(F, m, b):.2f} / {bms(F, m, b, 'p95_ms'):.2f}" for m in ("cnn", "vit")]] for b in backends]
    speed_cnn = bms(F, "cnn", "PyTorch CPU") / bms(F, "cnn", "ONNX Runtime CPU")
    slow_vit = bms(F, "vit", "ONNX Runtime CPU") / bms(F, "vit", "PyTorch CPU")
    n = F["nutrition"]
    nut_rows = [[k, v["serving"], v["calories"], v["protein"], v["carbs"], v["fat"]]
                for k, v in n["foods"].items() if k in ("samosa", "chicken_curry", "steak", "ice_cream")]
    return f"""
## 3.4 Phase 4: Nutrition and deployment {{#phase4}}

**What was done.**

* Added an approximate nutrition table.
* Exported the trained models to ONNX and **verified** them against PyTorch on every test image.
* Benchmarked PyTorch against ONNX Runtime.
* Built a FastAPI service and a Docker image that serve the ViT **without PyTorch**.

### Nutrition

`src/food_vision/resources/nutrition.json` maps every class to one *typical serving* with calories, protein, carbs and fat. Its disclaimer is returned in every API response:

> {n['disclaimer']}

{md_table(["Class", "Serving", "kcal", "Protein (g)", "Carbs (g)", "Fat (g)"], nut_rows)}

* **Code:** `nutrition.py` loads the file as typed dataclasses and reports classes without an entry.
* **Sanity test:** a unit test checks every entry for internal consistency using the Atwater factors: kcal ≈ 4×protein + 4×carbs + 9×fat, within 10%.
* **Be honest about these values:** they are rounded estimates written for this project, not looked up per item, and they are not measurements of the photo.

{box("concept", "Concept: ONNX and ONNX Runtime", """
* **ONNX** (Open Neural Network Exchange) is a framework-neutral file format: the model's computation graph plus its weights. Export once, and run anywhere: servers, mobile, browsers, edge devices.
* **ONNX Runtime (ORT)** is Microsoft's inference engine for ONNX files. It optimises the graph (fusing operations, folding constants) and runs it through an **execution provider (EP)**:
    * **CPUExecutionProvider:** the default.
    * **CoreMLExecutionProvider:** Apple's Neural Engine and GPU.
    * **CUDA / TensorRT EPs:** NVIDIA GPUs.
* **Partial support:** if an EP can't handle some operations, ORT splits the graph into partitions and runs the unsupported parts on CPU. Every hand-off between partitions costs time.""")}

### Export and verification

* **Exporter:** `export.export_onnx` uses PyTorch's `torch.export`-based exporter (`dynamo=True`) at opset 18, with a **dynamic batch dimension**.
* **Self-describing file:** the class names and preprocessing settings are written into the ONNX file's metadata, so the API needs nothing but that one file.
* **Validity check:** `onnx.checker` validates the graph.

`scripts/export_onnx.py` then runs **three independent checks** on all {ov['conversion']['num_images']} test images:

{md_table(["Check", "What it isolates", "ViT-S/16", "EfficientNet-B0"], [
    ["1. Conversion", "identical tensors into PyTorch and ORT: max absolute logit difference",
     f"{ov['conversion']['max_abs_logit_diff']:.1e}", f"{oc['conversion']['max_abs_logit_diff']:.1e}"],
    ["", "top-1 agreement PyTorch vs ORT", pct(ov['conversion']['top1_agreement']), pct(oc['conversion']['top1_agreement'])],
    ["2. Preprocessing", "NumPy/PIL vs torchvision eval transform: max abs difference",
     f"{ov['preprocessing_max_abs_diff']:.1f}", f"{oc['preprocessing_max_abs_diff']:.1f}"],
    ["3. End to end", "ORT + NumPy preprocessing top-1", pct(ov['end_to_end_onnx_top1']), pct(oc['end_to_end_onnx_top1'])],
    ["", "PyTorch fp32 top-1 (like-for-like)", pct(ov['conversion']['torch_top1']), pct(oc['conversion']['torch_top1'])],
    ["", "Phase 2 recorded top-1 (fp16 autocast)", pct(ov['recorded_pytorch_top1']), pct(oc['recorded_pytorch_top1'])],
    ["", "ONNX file size", f"{ov['onnx_size_mb']} MB", f"{oc['onnx_size_mb']} MB"],
])}

{box("why", "Why the end-to-end number is 0.05 points higher than Phase 2", "Phase 2 evaluated on MPS with fp16 autocast; these checks run in fp32. In fp32, PyTorch and ONNX agree exactly, and both get one more test image right than the fp16 run did. Precision changes predictions only on images right at a decision boundary. The verification report records this explicitly.")}

### Latency: PyTorch vs ONNX Runtime

`scripts/benchmark.py`: batch {env['batch_size']}, fp32, {env['warmup']} warm-up runs, then mean / p95 (ms) over {env['iters']} runs. Both backends use the same {env['cpu_threads']} CPU threads. Environment: torch {env['torch']}, onnxruntime {env['onnxruntime']}, Python {env['python']}, {env['machine']}.

{md_table(["Backend", "EfficientNet-B0 mean / p95", "ViT-S/16 mean / p95"], rows)}

**Two surprises, both explained**

1. **The CNN is {speed_cnn:.1f}× faster on ORT CPU than on PyTorch CPU.** This confirms the Phase 3 diagnosis: PyTorch's slow CPU number came from the missing oneDNN in this build, not from EfficientNet. ORT's CoreML EP runs it in about {bms(F, 'cnn', 'ONNX Runtime CoreML'):.1f} ms.
2. **The ViT is {slow_vit:.1f}× slower on ORT CPU, and even slower on CoreML.**
    * **CPU:** profiling showed about 70% of ORT's time in MatMul, so the ViT is matrix-multiply-bound. The *likely* reason PyTorch wins is that on macOS it uses Apple's Accelerate BLAS for matrix multiplies.
    * **CoreML:** the EP supported only part of the ViT graph (279 of 527 nodes, in 99 partitions), so execution keeps bouncing between CoreML and CPU.
    * **Provenance:** the profile and node counts come from the benchmarking session (noted in the README), not from a stored report file.

{box("interview", "The takeaway to say out loud", "The fastest backend depends on the architecture and the hardware, so you measure rather than assume. For this Mac, the depthwise-conv CNN is fastest on ORT (especially CoreML), and the GEMM-heavy ViT is fastest on PyTorch MPS. The deployment target (a Linux container) uses ORT CPU, so the CPU column is the relevant one there.")}

### The API (FastAPI)

{box("concept", "Concept: FastAPI, Uvicorn and Pydantic", """
* **FastAPI** maps Python functions to HTTP endpoints.
* **Pydantic** models define and validate the request and response shapes, and FastAPI uses them to generate interactive documentation at `/docs` automatically.
* **Uvicorn** is the ASGI server that actually serves the app.
* **Endpoint style:** `/predict` is a plain `def`, not `async def`. FastAPI runs such endpoints in a thread pool, so CPU-heavy inference doesn't block the event loop.""")}

* **Start-up:** a `lifespan` handler loads the ONNX model and nutrition table **once**. The model path, upload limit and thread count come from environment variables.
* **`POST /predict`:** returns the top-k dishes (default 3) with confidence, display name and nutrition, plus the disclaimer and the server-side inference time.
* **Errors:** unsupported type → **415**, empty or undecodable image → **400**, too large → **413**, bad `top_k` → **422** (automatically, from the `Query(ge=1, le=10)` constraint).
* **Preprocessing:** `inference.preprocess` re-implements torchvision's eval transform in NumPy/PIL, including torchvision's exact rounding rules. That is what keeps PyTorch out of the image.

{box("concept", "Concept: Docker", """
A **Docker image** packages the code, its dependencies and the model into one runnable unit, so it behaves identically on any machine.

* **Small base:** the Dockerfile starts from `python:3.12-slim` and installs only `requirements-api.txt`: FastAPI, Uvicorn, ONNX Runtime, NumPy and Pillow. **No PyTorch.**
* **Layer caching:** dependencies are installed before the code is copied, so code changes don't reinstall everything.
* **Security:** the container runs as a **non-root** user.
* **Health:** a **HEALTHCHECK** calls `/health`.
* **Minimal build context:** `.dockerignore` is an allow-list, so only the package, the serving requirements and the ONNX file reach the build.
* **Which model:** `ARG MODEL` selects which ONNX file to serve.""")}
"""


def sec_phase5(F: dict) -> str:
    return f"""
## 3.5 Phase 5: Quality and documentation {{#phase5}}

**What was done.**

* A pytest suite of **{F['n_tests']} tests** that runs in seconds **without the dataset or any trained model**.
* The README.
* Two notebooks: an executed results walkthrough, and a Colab training notebook.

{box("concept", "Concept: pytest fixtures", "A **fixture** is reusable setup that pytest injects into the tests that ask for it by name. `tests/conftest.py` builds a **fake Food-101 tree** (3 used classes plus one distractor, solid-colour JPEGs) and a **tiny real ONNX classifier** made with `onnx.helper` (global average pool → flatten → linear). The data and API code therefore run against real files and a real ONNX Runtime session, just very small ones.")}

{md_table(["Test file", "What it protects"], [
    ["`test_transforms.py`", "transform shapes and determinism; **NumPy preprocessing == torchvision** for landscape, portrait, small, greyscale, RGBA and rounding edge-case sizes"],
    ["`test_data.py`", "split sizes, no overlap, test images come from the official test list, sorted labels, determinism, exclude list, fingerprint rebuilds, bad configs rejected"],
    ["`test_nutrition.py`", "every configured class has an entry, Atwater consistency, unknown class → None, disclaimer present, negative values rejected"],
    ["`test_api.py`", "/health, /classes, /predict (sorted top-k, nutrition, PNG with alpha), and 400 / 413 / 415 / 422 error codes"],
    ["`test_metrics.py`", "top-k accuracy, confusion matrix, per-class report, confused pairs, McNemar"],
    ["`test_engine.py`", "warmup + cosine LR values, early-stopping logic, config loading and validation"],
    ["`test_detection.py` (Phase 6)", "letterbox (incl. odd-padding rounding), IoU, NMS, class-aware decode, box mapping, food filtering, fallback, the /detect endpoint (added later; see 3.6)"],
])}

{box("why", "Mutation checks: testing the tests", "To check the parity test would actually catch a bug, the preprocessing was deliberately broken three ways: floor instead of round for the vertical crop offset, the same for the horizontal offset, and bicubic instead of bilinear resizing. At first the rounding mutants **passed**: Python's `round` uses banker's rounding, so `round(x/2)` and floor division only differ when the offset ends in .5 and rounds up, and no test image size triggered that. Sizes such as 309×300 (offset 19.5) were added; now all three mutants fail. This is a strong interview story about test quality.")}

**Documentation**

* **README:** the problem, a mermaid architecture diagram, results and significance tables, error analysis, ONNX and benchmark results, API usage and the v2 roadmap.
* **`notebooks/results_walkthrough.ipynb`:** loads the recorded outputs and shows every result inline, without retraining. It renders on GitHub.
* **`notebooks/colab_train.ipynb`:** runs the same scripts on a Colab GPU (with an option for ViT-B/16). Its outputs go to separate folders, so they can't overwrite the recorded results. It has not been run on Colab yet.
"""


def sec_detection(F: dict, figs: dict) -> str:
    y = F["yolo"]
    so, pr = y["comparisons"]["same_onnx"], y["comparisons"]["pytorch_rect"]
    thr = float(y["pass_criterion"].split(">=")[1].split()[0])
    labels = ", ".join(f"`{n}`" for n in F["food_labels"])
    ver_rows = [[name, f"{c['reference_detections']} / {c['onnx_detections']}", c["matched"],
                 f"{pct(c['match_rate_vs_reference'])} / {pct(c['match_rate_vs_onnx'])}",
                 f"{c['mean_iou_matched']:.4f}", f"{c['max_abs_score_diff_matched']:.3f}"]
                for name, c in (("Ultralytics, same ONNX file (**pass/fail**)", so),
                                ("Ultralytics, PyTorch weights (info)", pr))]
    demo = F.get("demo")
    if demo and figs.get("detect"):
        lines = []
        if "multi" in demo:
            d = demo["multi"]
            regs = "; ".join(f"a `{lab}` region ({s:.2f}) classified as **{c}** ({pct(p, 1)})"
                             for lab, s, c, p in d["dishes"])
            lines.append(f"* **Left, `{d['path']}`:** {len(d['dishes'])} regions: {regs}.")
        if "fallback" in demo:
            d = demo["fallback"]
            _, _, c, p = d["dishes"][0]
            lines.append(f"* **Right, `{d['path']}`:** no food region detected, so the whole image is classified: "
                         f"**{c}** ({pct(p, 1)}).")
        demo_md = (fig(figs["detect"], "Real output of detect_and_classify, generated when this PDF was built "
                       "(boxes from YOLO11n, labels from the ViT-S/16).", 100)
                   + "\n".join(lines) + "\n")
    else:
        demo_md = "<p class=\"small\">(Demo figure omitted: export `models/yolo11n.onnx` and `models/vit.onnx` to include it.)</p>\n"
    return f"""
## 3.6 Phase 6: Multi-food detection (YOLO) {{#phase6}}

**What was done.**

* Added a `POST /detect` endpoint that finds **every** food region in a photo and labels each with a dish, a confidence score and nutrition.
* `/predict` is unchanged: same code path, same response, still tested.
* The detector is a COCO-pretrained **YOLO11n** exported to ONNX ({y['onnx_size_mb']} MB). Serving it needs **no new packages**: no PyTorch, no Ultralytics, no OpenCV.

### Why detect-then-classify (two stages)

* **No boxes in Food-101:** the dataset has one label per image and no bounding boxes, so there is nothing here to train or evaluate a food detector on.
* **Reuse what's verified:** the ViT-S/16 already has measured accuracy and a verified ONNX export. A generic detector only has to answer *"where is there food?"*; the classifier answers *"which dish?"*.
* **The pipeline** (`detection.detect_and_classify`):
    1. YOLO proposes regions (confidence ≥ {y['conf']}, NMS IoU {y['iou']}).
    2. Each region is cropped with 5% context on every side and classified by the ViT. Boxes smaller than 16 px on either side are skipped.
    3. **Fallback:** if no food region is found (common for close-up photos), the whole image is classified, which gives exactly the `/predict` result. A unit test checks this equality.
    4. The API joins every dish with `nutrition.json`.

{box("why", "Why not train a food detector?", "Training needs box- or mask-labelled food data (e.g. UECFOOD-256), and evaluating it needs mAP on a held-out labelled set. Neither exists in this repo, so **no detection accuracy or mAP is claimed**. The two-stage design is the honest version: it adds multi-dish capability while every number it reports is measured.")}

### The COCO class-filtering decision

COCO has 80 classes, and only some are food. The detector keeps only these proposal classes: {labels}.

* **Prepared dishes** (pizza, sandwich, hot dog, donut, cake) map naturally onto dishes.
* **`bowl`** is included because curries, ramen, salads and rice are usually served in one.
* **Raw produce is excluded** (banana, apple, orange, broccoli, carrot):
    * **The reason:** the dish classifier is closed-set. It must pick one of its {len(F['classes'])} dishes and has no class for raw produce, so a garnish crop is forced into a wrong label.
    * **The evidence:** during development, carrot garnish on a steak plate was detected, cropped and classified as samosa. That was observed in a demo and isn't stored in a report file.
* **Configurable:** the list is a constructor argument (`food_labels`), so a deployment with a produce class could add those labels back.

### Re-implementing YOLO pre/post-processing in NumPy

The ONNX file outputs raw predictions: a tensor of shape (1, 84, 8400). That is 8,400 candidate boxes, each with 4 box coordinates (centre x, centre y, width, height) and 80 class scores. Ultralytics normally does the pre- and post-processing in PyTorch, so `detection.py` re-implements it:

* **Letterbox:** resize keeping the aspect ratio so the long side is 640, then pad to 640×640 with grey (114).
    * **Copied arithmetic:** the padding uses Ultralytics' exact rule `round(d - 0.1)`. For odd padding this decides whether the extra pixel goes on the left or the right; a wrong choice shifts every box by one pixel.
* **Decode:** for every candidate, take the best class and its score, keep it if the score > conf, and convert centre/size to corner coordinates (x1, y1, x2, y2).
* **Class-aware NMS** (non-maximum suppression):
    * **The trick:** offset each box by `class_id × 7680` (the same constant Ultralytics uses), then run one greedy NMS. Boxes of different classes can then never overlap, so they never suppress each other.
    * **Greedy NMS:** keep the highest-scoring box, drop the remaining boxes whose IoU with it exceeds the threshold, and repeat.
* **Un-letterbox:** subtract the padding, divide by the scale and clip to the image, giving boxes in original-image pixels.
* **Self-describing model:** the class names and input size are read from the ONNX metadata that Ultralytics writes. A model without that metadata, or an end-to-end export with built-in NMS, is rejected with a clear error.

{box("concept", "Concept: IoU and NMS", "**IoU** (intersection over union) is the overlap area of two boxes divided by the area of their union: 1 for identical boxes, 0 for disjoint ones. A detector fires many near-duplicate boxes on each object, and **NMS** keeps the best one per object by suppressing lower-scoring boxes that overlap it above an IoU threshold.")}

### Verification against Ultralytics

`scripts/export_yolo.py` exports the model, then runs both pipelines on **{y['num_images']} random test images** (seeded). It matches detections one-to-one (same class, IoU ≥ {y['match_iou']}) in both directions and fails with exit code 1 if either match rate is below {pct(thr, 0)}.

{md_table(["Reference", "Detections (reference / ours)", "Matched", "Match rate (vs reference / vs ours)", "Mean IoU", "Max score diff"], ver_rows)}

* **Result: {"PASSED" if y['passed'] else "FAILED"}.** The remaining differences against the same ONNX file come from resizing: the NumPy pipeline uses PIL bilinear, while Ultralytics uses OpenCV. PIL keeps OpenCV out of the API.
* **Coverage:** {y['images_with_food_detection']} of the {y['num_images']} images get at least one food region. The rest use the whole-image fallback.

{box("caveat", "The rectangular-vs-640×640 lesson", f"""
* **The first version of this check FAILED:** it compared against Ultralytics running the original PyTorch weights, and agreement was {pct(pr['match_rate_vs_reference'])} / {pct(pr['match_rate_vs_onnx'])}, below the {pct(thr, 0)} threshold, with score differences up to {pr['max_abs_score_diff_matched']:.2f}.
* **The cause was the reference, not the code:** Ultralytics runs `.pt` weights with **rectangular inference**. It pads only to a multiple of the stride (e.g. 640×480), whereas the static ONNX export always sees a full 640×640 letterbox. A different input gives slightly different detections.
* **The fix:** compare against Ultralytics running the **same ONNX file**. Both pipelines then feed identical input shapes to the identical model, which isolates exactly what was re-implemented (letterbox, decode, NMS). Agreement rose to {pct(so['match_rate_vs_reference'])} / {pct(so['match_rate_vs_onnx'])}.
* **The PyTorch comparison is kept** as an informational row, because it measures something real: the cost of a static export.
* **The general lesson:** when a verification fails, first check that both sides receive the same input. Choose a reference that isolates the component you changed.""")}

### Demo

{demo_md}
### API, Docker and tests

* **`POST /detect?top_k=3&conf=0.25`:** for each dish it returns the box (original-image pixels), `source` (`detection` or `whole_image`), the detector's COCO label and score, and the top-k dishes with nutrition. It also returns the image size, the disclaimer and the timing.
* **Shared validation:** the same 415 / 400 / 413 checks as `/predict`, via one shared upload helper. `conf` and `top_k` are range-checked (422).
* **Optional detector:** if `FOOD_VISION_DETECTOR` (default `models/yolo11n.onnx`) is missing, the API still starts, `/predict` works and `/detect` returns **503**.
* **Docker:** a multi-stage build copies the detector only if it was exported, so an image without it is unchanged. Neither image contains PyTorch or Ultralytics.
* **Tests:** {F['n_det_tests']} tests in `test_detection.py`. A fixture builds a **tiny real ONNX "detector"** with Ultralytics-style metadata and fixed anchors (a person, a bowl plus a near-duplicate bowl, a low-score pizza), so NMS, food filtering, thresholds and box mapping are all tested exactly.
* **Mutation checks:** the letterbox rounding, class-agnostic NMS and the unletterbox maths were each broken on purpose.
    * **The first catch:** the rounding mutant initially **survived**, because of banker's rounding again: a padding of 10.5 rounds the same way under both rules.
    * **The fix:** sizes such as 100×64 (padding 11.5) were added. Every mutant now fails the tests.

### Known weaknesses

* **No accuracy measurement:** there is no mAP and no per-dish accuracy for multi-dish photos.
* **Coarse proposals:** COCO food classes are coarse. A samosa is proposed as a `sandwich`, and anything that isn't one of the 6 classes is only found via the fallback.
* **Closed-set classifier:** it can't say "none of these". A small sauce bowl still gets a dish label (in the demo: samosa at low confidence).
* **Crops are out of distribution:** the ViT was trained on whole Food-101 photos, not tight crops.
* **Licence:** Ultralytics code and the YOLO weights are **AGPL-3.0**. The served detector is derived from them, so the repository is now licensed AGPL-3.0 (`LICENSE`, plus a Licence section in the README). Permissive alternatives: RT-DETR, torchvision detectors, or an Ultralytics Enterprise licence.
"""


def sec_deep_dive(F: dict) -> str:
    t, c = F["test"], F["comp"]
    vc = F["sig"].query("model_a=='vit' and model_b=='clip'").iloc[0]
    sp, ic = F["steak_pair"], F["icecream_pair"]
    gap = F["val_test_gap"]
    vit_err = int(c.loc["vit", "errors"])
    return f"""
# 4. Results deep-dive {{#deepdive}}

## 4.1 Why zero-shot CLIP scored highest, and why that's not "better"

CLIP reached **{pct(t['clip']['top1'])}** top-1 without a single training image, against **{pct(t['vit']['top1'])}** for the fine-tuned ViT. Three reasons explain why it does so well:

* **Huge pretraining:** LAION-2B pretraining almost certainly included many food photos with captions naming the dish.
* **Common names:** the 20 class names are common words the text encoder understands well.
* **Robust class embeddings:** prompt ensembling averages out the wording of any single prompt.

**But the gap is not significant.** The two models disagree on {vc.only_a_correct + vc.only_b_correct} test images: the ViT alone is right on {vc.only_a_correct} and CLIP alone on {vc.only_b_correct}. If the models were equal, a split this uneven would happen by chance with p = {vc.p_value:.2f}. **Say this precisely in an interview:** "CLIP and the ViT are statistically tied; both beat the CNN."

## 4.2 Why ViT-S/16 is the deployed model

{md_table(["", "ViT-S/16", "CLIP ViT-B/32"], [
    ["Test top-1", pct(t['vit']['top1']), pct(t['clip']['top1'])],
    ["Significant difference?", "no (p = %.2f)" % vc.p_value, ""],
    ["Inference parameters", f"{c.loc['vit', 'params_inference'] / 1e6:.1f}M", f"{c.loc['clip', 'params_inference'] / 1e6:.1f}M"],
    ["Latency MPS (mean)", f"{c.loc['vit', 'mps_mean_ms']:.1f} ms", f"{c.loc['clip', 'mps_mean_ms']:.1f} ms"],
    ["Latency CPU (mean)", f"{c.loc['vit', 'cpu_mean_ms']:.1f} ms", f"{c.loc['clip', 'cpu_mean_ms']:.1f} ms"],
    ["ice cream ↔ frozen yogurt errors", ic['vit'].total, ic['clip'].total],
])}

At equal accuracy, the smaller and faster model wins. The ViT is also *trained on this exact task*, so it can be improved with more labelled data in a way zero-shot prompts can't. CLIP remains valuable for **adding new dishes without training data**, and as a candidate for fine-tuning (see the interview questions).

## 4.3 steak vs filet mignon

* **The dominant error:** this pair is the top confusion for every model: {sp['cnn'].total} errors for the CNN, {sp['vit'].total} for the ViT and {sp['clip'].total} for CLIP. For the ViT, that is {pct(sp['vit'].total / vit_err, 0)} of its {vit_err} test errors.
* **Why it's hard:** it is a **fine-grained** distinction. Filet mignon *is* a cut of steak, and once cooked and plated the visual cues are subtle (thickness, round shape, often wrapped in bacon). Many Food-101 images also show it sliced or partly hidden.
* **Possible fixes** are covered in the interview section: more targeted data, higher input resolution, hierarchical labels, or merging the classes if the product doesn't need the distinction.

## 4.4 Why test accuracy is higher than val accuracy

Every model scores higher on test than on val: by {100 * gap['cnn']:.1f} points (CNN), {100 * gap['vit']:.1f} points (ViT) and {100 * gap['clip']:.1f} points (CLIP).

**The key clue is CLIP.** It never trained on or tuned against either split, so selection bias or overfitting to val can't explain its gap. The explanation is how Food-101 was built: val is sampled from the official **train** images, which were left uncleaned with some wrong labels, while the official **test** images were manually cleaned. Noisier labels in val mean lower measured accuracy.

## 4.5 The duplicates found and removed

The EDA hashed every image file. Two problems were found, and both are now in the config's `exclude` list with comments:

{chr(10).join(f"* `{img}`: {why}" for img, why in F['exclude'])}

Exclusion happens *after* shuffling, so only those slots change, and each class is refilled from its unused pool, keeping exactly {F['cfg']['data']['train_per_class']} / {F['cfg']['data']['val_per_class']} / {F['cfg']['data']['test_per_class']} per class.

## 4.6 The benchmark surprises

* **Missing oneDNN:** this PyTorch build reports oneDNN unavailable, so EfficientNet's depthwise convolutions take a slow fallback path on CPU ({bms(F, 'cnn', 'PyTorch CPU'):.1f} ms). ONNX Runtime runs the same graph in {bms(F, 'cnn', 'ONNX Runtime CPU'):.2f} ms.
* **ViT slower on ORT:** the ViT is matrix-multiply-bound. On this Mac, PyTorch's CPU path ({bms(F, 'vit', 'PyTorch CPU'):.2f} ms) beats ORT CPU ({bms(F, 'vit', 'ONNX Runtime CPU'):.2f} ms), most likely because of Apple's Accelerate BLAS.
* **ViT even slower on CoreML** ({bms(F, 'vit', 'ONNX Runtime CoreML'):.2f} ms): the graph is only partly supported and is split into many CPU↔CoreML partitions.
* **Lesson:** the fastest backend depends on the architecture and the hardware, so benchmark on the real target.
"""


def sec_code_tour(F: dict) -> str:
    files = [
        ("src/food_vision/config.py", "Typed dataclasses (`DataConfig`, `TrainConfig`, `ClassifierConfig`, `ClipConfig`) loaded from YAML. Rejects unknown keys (catches typos). `split_fingerprint` hashes every setting that affects the split."),
        ("src/food_vision/data.py", "Resumable download + MD5 + safe extraction; `build_splits` (deterministic, leak-free); CSV manifests with fingerprint; `FoodDataset`; `build_transforms`; DataLoaders."),
        ("src/food_vision/device.py", "`get_device`: MPS → CUDA → CPU, so the same code runs on a Mac and on Colab."),
        ("src/food_vision/utils.py", "`seed_everything` (Python, NumPy, torch) and `seed_worker` for reproducible DataLoader workers."),
        ("src/food_vision/models.py", "`build_classifier` (torchvision or timm, new head, returns the model's preprocessing); `save_checkpoint` / `load_classifier` store arch, classes and preprocessing with the weights."),
        ("src/food_vision/engine.py", "Mixed-precision setup, AdamW param groups, warmup+cosine schedule, `train_one_epoch`, `predict` / `evaluate`, `EarlyStopping`, `CSVLogger`."),
        ("src/food_vision/metrics.py", "`topk_accuracy` and `save_test_outputs` (writes `test_predictions.npz` + `test_metrics.json` for every model in the same format)."),
        ("src/food_vision/clip_zero_shot.py", "`ZeroShotCLIP`: OpenCLIP model + prompt-ensembled text-embedding matrix; outputs logits like a classifier."),
        ("src/food_vision/evaluation.py", "Works on saved logits: `Predictions`, `confusion_matrix`, `per_class_report`, `most_confused_pairs`, `worst_errors`, `mcnemar_exact`."),
        ("src/food_vision/benchmark.py", "`measure_latency` (warm-up, device sync, mean/p50/p95) and `torch_latency`; reused by the PyTorch-vs-ONNX benchmark."),
        ("src/food_vision/explain.py", "`GradCAM`, `AttentionRollout`, `artifact_token_check`, plus overlay helpers."),
        ("src/food_vision/export.py", "`export_onnx` (dynamic batch, metadata, checker) and `compare_logits` (agreement statistics)."),
        ("src/food_vision/inference.py", "Torch-free serving path: `preprocess` (torchvision-exact), `softmax`, `OnnxClassifier` (reads classes and preprocessing from the ONNX metadata)."),
        ("src/food_vision/nutrition.py", "`load_nutrition` / `default_nutrition`, typed `Nutrition` entries, `missing()` coverage check."),
        ("src/food_vision/detection.py", "`OnnxYoloDetector` (letterbox, decode, class-aware NMS, unletterbox in NumPy), `detect_and_classify` (crop each food region → classifier, whole-image fallback), `draw_dishes`."),
        ("src/food_vision/api/app.py, schemas.py", "FastAPI app (lifespan loading, `/health`, `/classes`, `/predict`, `/detect`, shared upload validation) and Pydantic response models."),
        ("scripts/prepare_data.py", "Download + build splits; prints per-class counts and asserts the splits don't overlap."),
        ("scripts/eda.py", "Class counts, per-image stats, decode check, MD5 duplicates, sample grid, `summary.json`."),
        ("scripts/train.py", "Fine-tunes `--model cnn` or `vit`, logs `metrics.csv`, keeps `best.pt`, evaluates the best checkpoint on test."),
        ("scripts/zero_shot_clip.py", "Evaluates CLIP on val and test with its own preprocessing; same output files as training."),
        ("scripts/evaluate.py", "Per-class reports, confusion matrices, confused pairs, worst errors, all-wrong images, McNemar, latency table, `summary.md`."),
        ("scripts/explain.py", "Grad-CAM vs rollout figures on the same images + the artifact-token check."),
        ("scripts/export_onnx.py", "Export + the three verification checks on the full test split; non-zero exit code if they fail."),
        ("scripts/benchmark.py", "PyTorch CPU/MPS vs ONNX Runtime CPU/CoreML latency → `reports/deploy/benchmark.*`."),
        ("scripts/export_yolo.py", "Exports YOLO11n to ONNX and verifies the NumPy pipeline against Ultralytics → `reports/deploy/yolo_verify.json`; non-zero exit code if it fails."),
        ("scripts/detect.py", "CLI: draws boxes + dish labels on images and prints each dish with kcal."),
        ("scripts/build_study_guide.py", "Builds this PDF from the recorded files (personal tooling)."),
        ("tests/conftest.py", "Fixtures: fake Food-101 tree, config factory, tiny ONNX classifier, tiny ONNX YOLO-style detector, image-bytes helper."),
        ("tests/test_*.py", "See section 3.5 for what each test file protects."),
    ]
    tour = md_table(["File", "What it does"], [[f"`{f}`", d] for f, d in files])

    ex = []
    ex.append(("Data splits: deterministic and leak-free", excerpt("src/food_vision/data.py", "build_splits"),
               """
* **Per-class generator:** `random.Random(f"{seed}-{name}")` gives each class its own seeded generator, so classes don't affect each other.
* **Sorted before shuffling:** the pool is sorted *before* the shuffle, so the order of the JSON file can't change the result.
* **No overlap by construction:** val takes the first `n_val` images of the shuffled train pool and train takes the next ones, so the two can't overlap. Test only ever reads the official test list.
* **Exclusions:** they are filtered after the shuffle, so removing one image doesn't reshuffle the rest."""))
    ex.append(("Training loop with mixed precision", excerpt("src/food_vision/engine.py", "train_one_epoch"),
               """
* **Autocast:** wraps only the forward pass and the loss.
* **Safety check:** a non-finite loss stops training with an actionable message.
* **Order of operations with the GradScaler:** scale the loss, run backward, **unscale** (so gradient clipping sees the true gradient values), clip, then step, then update the scale factor.
* **Schedule:** `scheduler.step()` runs every batch (per-step warmup + cosine)."""))
    ex.append(("Warmup + cosine schedule", excerpt("src/food_vision/engine.py", "warmup_cosine"),
               "* `LambdaLR` multiplies the base learning rate by `factor(step)`: linear warmup to 1.0, then a half cosine down to `min_ratio`."))
    ex.append(("Zero-shot CLIP class embeddings", excerpt("src/food_vision/clip_zero_shot.py", "ZeroShotCLIP"),
               """
* **Building the matrix:** each class's prompts are embedded, L2-normalised, averaged and re-normalised. The result is a single (embedding dim × classes) matrix.
* **Forward pass:** one image encoding and one matrix multiply, scaled by CLIP's learned temperature."""))
    ex.append(("McNemar's exact test", excerpt("src/food_vision/evaluation.py", "mcnemar_exact"),
               "* Only the discordant counts matter. The p-value is a two-sided binomial tail with probability 0.5."))
    ex.append(("Grad-CAM", excerpt("src/food_vision/explain.py", "GradCAM.__call__"),
               "* The forward hook (not shown) stores the activations and registers a tensor hook that captures their gradients. Channel weights are the spatially averaged gradients; the map is ReLU of the weighted sum, upsampled."))
    ex.append(("Attention rollout", excerpt("src/food_vision/explain.py", "AttentionRollout.__call__"),
               "* `self._attn` holds the post-softmax attention matrices captured by hooks during the forward pass. Average over heads, add the identity, re-normalise rows, multiply across layers, then read the [CLS] row over the patch tokens."))
    ex.append(("ONNX export", excerpt("src/food_vision/export.py", "export_onnx"),
               """
* **Dynamic batch:** `dynamic_shapes` makes the batch dimension variable (1–1024).
* **Single file:** `external_data=False` keeps the weights inside the one `.onnx` file.
* **Metadata:** the class names and preprocessing are written into `metadata_props` and validated with `onnx.checker`."""))
    ex.append(("ONNX verification (scripts/export_onnx.py)",
               excerpt_between("scripts/export_onnx.py", "# 1. Conversion check", "recorded = json.loads"),
               "* Check 1 feeds *identical tensors* to both runtimes (isolating the conversion). Checks 2 and 3 run the real serving path (NumPy preprocessing → ORT) on every test image."))
    ex.append(("API preprocessing parity", excerpt("src/food_vision/inference.py", "preprocess"),
               "* This mirrors torchvision exactly: `int()` truncation for the long side, `int(round(...))` for crop offsets (Python's banker's rounding, same as torchvision), bilinear resampling via Pillow."))
    ex.append(("The test that enforces it", excerpt("tests/test_transforms.py", "test_numpy_preprocess_matches_torchvision"),
               "* Parametrised over shapes and colour modes. A failing case would mean the deployed model sees different pixels than the model was trained and evaluated on."))
    ex.append(("The /predict endpoint", excerpt("src/food_vision/api/app.py", "predict"),
               "* Validation (in the shared `_read_image` helper) happens before any expensive work. `image.load()` forces a full decode, so truncated files fail with a 400 rather than later inside the model."))
    ex.append(("YOLO letterbox (detection.py)", excerpt("src/food_vision/detection.py", "letterbox"),
               "* `round(d - 0.1)` is Ultralytics' padding rule: for odd padding the extra pixel goes right/bottom. A test with 11.5 px padding pins it down (10.5 can't, because of banker's rounding)."))
    ex.append(("Decode + class-aware NMS", excerpt("src/food_vision/detection.py", "decode"),
               "* Transposing gives one row per candidate box. The `class_ids * MAX_WH` offset moves each class into its own far-apart region of coordinate space, so one NMS call never lets a bowl suppress a pizza."))
    ex.append(("Detect, crop, classify", excerpt("src/food_vision/detection.py", "detect_and_classify"),
               "* Each crop goes through the same `OnnxClassifier.predict` as `/predict`, so preprocessing parity carries over. With no regions, the fallback is literally the `/predict` computation."))

    parts = [f"""
# 5. Code tour {{#codetour}}

**How the pieces connect:**

* **Config:** `config.py` is read by every script.
* **Data and training:** `data.py` and `models.py` feed `engine.py` (training) and `metrics.py` (uniform test outputs).
* **Evaluation:** `evaluation.py` and `explain.py` consume those outputs.
* **Serving:** `export.py` turns a checkpoint into ONNX. `inference.py` is the torch-free runtime used by `api/app.py`, which joins predictions with `nutrition.py`.
* **Detection:** `detection.py` wraps the YOLO ONNX file and reuses `inference.OnnxClassifier` for every crop.

{tour}

## Annotated excerpts

These excerpts are extracted from the source files when this PDF is built, so they always match the code.
"""]
    for title, src, notes in ex:
        parts.append(f"### {title}\n\n{code(src)}\n{notes}\n")
    return "\n".join(parts)


def sec_how_to_run(F: dict) -> str:
    return f"""
# 6. How to run everything {{#howtorun}}

**Setup** (Python 3.10+; the project was developed on Python {F['bench_env']['python']}):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**The pipeline, in order:**

{md_table(["Step", "Command", "Output"], [
    ["1. Data", "`python scripts/prepare_data.py`", "Food-101 in `data/` (~5 GB, resumable), `data/splits/*.csv`"],
    ["2. EDA", "`python scripts/eda.py`", "`reports/eda/`"],
    ["3. CLIP zero-shot", "`python scripts/zero_shot_clip.py`", "`runs/clip/`"],
    ["4. Train CNN", "`python scripts/train.py --model cnn`", "`runs/cnn/`"],
    ["5. Train ViT", "`python scripts/train.py --model vit`", "`runs/vit/`"],
    ["6. Evaluate", "`python scripts/evaluate.py`", "`reports/eval/` (`--skip-latency` to skip timing)"],
    ["7. Explain", "`python scripts/explain.py`", "`reports/explain/`"],
    ["8. Export + verify", "`python scripts/export_onnx.py --model vit`", "`models/vit.onnx`, `reports/deploy/onnx_verify_vit.json`"],
    ["9. Benchmark", "`python scripts/benchmark.py`", "`reports/deploy/benchmark.md`"],
    ["10. Detector export + verify", "`python scripts/export_yolo.py`", "`models/yolo11n.onnx`, `reports/deploy/yolo_verify.json` (needs `ultralytics`)"],
    ["11. Tests", "`pytest`", f"{F['n_tests']} tests, a few seconds, no dataset needed"],
])}

**Useful flags:**

* `train.py --epochs 1 --limit-batches 5 --runs-dir /tmp/x`: a smoke test in seconds.
* Edit `configs/default.yaml` to change the classes, images per split or models (for example ViT-B/16).

**Run the API:**

```bash
uvicorn food_vision.api.app:app --port 8000      # Swagger UI at http://localhost:8000/docs
curl -F "file=@my_lunch.jpg" "http://localhost:8000/predict?top_k=3"
curl -F "file=@my_lunch.jpg" "http://localhost:8000/detect?top_k=3&conf=0.25"   # one entry per dish
python scripts/detect.py my_lunch.jpg          # annotated image in reports/detect/
```

Environment variables: `FOOD_VISION_MODEL` (default `models/vit.onnx`), `FOOD_VISION_DETECTOR` (default `models/yolo11n.onnx`; optional, `/detect` returns 503 without it), `FOOD_VISION_MAX_UPLOAD_MB` (default 10), `FOOD_VISION_THREADS`.

**Docker** (export the model first):

```bash
docker build -t food-vision .                                  # serves models/vit.onnx
docker build -t food-vision-cnn --build-arg MODEL=models/cnn.onnx .
docker run --rm -p 8000:8000 food-vision
```

**Notebooks:** open them from the file Explorer, not from a chat link (in Antigravity, links open as raw JSON).

* `notebooks/results_walkthrough.ipynb`: shows all the results; nothing to run.
* `notebooks/colab_train.ipynb`: trains on a Colab GPU. Add a `GITHUB_TOKEN` secret while the repo is private.

**Rebuild this guide:** `python scripts/build_study_guide.py`
"""


def sec_limitations() -> str:
    return """
# 7. Limitations and roadmap {{#limitations}}

## 7.1 Honest caveats

* **Small scope:** 20 of 101 classes and 300 training images per class. Accuracy on the full 101-class problem would be lower and isn't measured here.
* **One run per model:** there are no repeated seeds, so run-to-run variance is unknown. The McNemar test covers test-set sampling, not training randomness.
* **One image, one label (classifier):** real meals often contain several dishes, and many of the "errors" are multi-dish photos. `/detect` now finds multiple dishes, but with a generic COCO detector and no measured detection accuracy (section 3.6).
* **Nutrition is per typical serving, not per photo:** there is no portion estimation, and the values are approximate estimates written for this project.
* **Explanations are approximate:** Grad-CAM is coarse (a 7×7 map upsampled), and plain attention rollout is distorted by artifact tokens, which this project demonstrated.
* **Benchmarks are from one Mac:** latency on a Linux server, a GPU or an edge device must be measured there. The Docker image itself was not formally benchmarked.
* **Training-time hardware:** MPS with fp16 autocast. Recorded fp16 results differ from fp32 by one test image per model.
* **No production hardening:** there's no authentication, rate limiting, monitoring or drift detection in the API.
* **The Colab notebook has not been executed on Colab yet.**

## 7.2 v2 roadmap

* **Food-specific detection and segmentation:** detect-then-classify with a COCO YOLO is done (section 3.6). Next: fine-tune YOLO (detection or instance segmentation) on box- or mask-labelled food data such as UECFOOD-256, and measure mAP and per-dish accuracy. Add an "unknown / not food" option to the classifier.
* **Portion estimation:** segmentation masks give each dish's area. With a size reference (the plate diameter, or depth from a phone's sensor), estimate volume, then grams, then nutrition per photo instead of per typical serving.
* **Edge deployment with TensorRT / INT8:** convert the ONNX model to a TensorRT engine for NVIDIA Jetson.
    * **FP16** is nearly free in accuracy.
    * **INT8** needs a calibration set of representative images, and must be validated with the same verification approach used for ONNX (agreement and accuracy on the test split).
* **Fine-grained classes:** collect targeted data for steak vs filet mignon and ice cream vs frozen yogurt, try a higher input resolution, or use hierarchical labels.
* **Better foundation-model use:** a linear probe or fine-tuning on CLIP's image encoder. It already matches the ViT zero-shot, so it may beat it once trained.
* **Explainability:** ViTs with register tokens, or gradient-weighted attention methods.
* **Meal recording:** store predictions per user and meal (the third part of the original paper), with user corrections feeding back as new labels.
* **MLOps:** CI running the tests on every push, model versioning, monitoring of prediction confidence and drift.
"""


def sec_interview(F: dict) -> str:
    t, c = F["test"], F["comp"]
    vc = F["sig"].query("model_a=='vit' and model_b=='clip'").iloc[0]
    sp, ic = F["steak_pair"], F["icecream_pair"]
    ov = F["onnx"]["vit"]
    art = max(F["artifact"], key=lambda a: a["mean_topk_overlap"])
    qa = [
        ("Walk me through the project in one minute.",
         f"It's a rebuild of my final-year project on food recognition and nutrition. I took a {len(F['classes'])}-class subset of Food-101 with leak-free, reproducible splits, then compared three approaches on identical data: a fine-tuned EfficientNet-B0, a fine-tuned ViT-S/16 and zero-shot CLIP. They reached {pct(t['cnn']['top1'])}, {pct(t['vit']['top1'])} and {pct(t['clip']['top1'])} top-1. McNemar tests show CLIP and the ViT are statistically tied and both beat the CNN. I did error analysis and explainability, exported the ViT to ONNX with verified 100% top-1 agreement, benchmarked runtimes, and served it with FastAPI in a Docker image without PyTorch. I then added multi-dish detection: a COCO YOLO proposes food regions, and the ViT classifies each crop. The suite has {F['n_tests']} tests."),
        ("Why not just use CLIP? It scored highest.",
         f"Its lead isn't statistically significant (p = {vc.p_value:.2f}). The ViT has {c.loc['vit', 'params_inference'] / 1e6:.1f}M inference parameters against CLIP's {c.loc['clip', 'params_inference'] / 1e6:.1f}M, and is faster: {c.loc['vit', 'cpu_mean_ms']:.1f} vs {c.loc['clip', 'cpu_mean_ms']:.1f} ms on CPU. CLIP is also clearly worse on ice cream vs frozen yogurt ({ic['clip'].total} vs {ic['vit'].total} confusions), where labelled data beats prompts. I'd keep CLIP for cold-starting new dishes without data, and I'd try fine-tuning or a linear probe on CLIP's image encoder; that could beat both."),
        ("How did you make sure there's no data leakage?",
         "Train and val are sampled without overlap from Food-101's official train list, and test only from the official test list, so test images are never trained on. The split is deterministic and versioned by a config fingerprint. The EDA hashes every file, which caught a train/val duplicate that I excluded. The checkpoint is chosen on val and evaluated on test exactly once."),
        ("Why is test accuracy higher than validation accuracy?",
         f"All three models show it, by {100 * F['val_test_gap']['clip']:.1f}–{100 * F['val_test_gap']['cnn']:.1f} points, including zero-shot CLIP, which never touched either split. So it isn't overfitting to val. Food-101's training images were intentionally left noisy, while the test images were manually cleaned, and my val set comes from the training images."),
        ("How would you improve steak vs filet mignon?",
         f"It's the top confusion for every model ({sp['vit'].total} errors for the ViT). First, check whether the product even needs the distinction; merging the classes or using a hierarchical 'steak → cut' label may be the right answer. If it does: collect more targeted, cleanly labelled data; use higher input resolution, since the cues are thickness and shape; consider fine-grained techniques or a stronger backbone; and use per-class confidence thresholds to abstain on ambiguous images. I'd measure each change on that pair's recall, not just overall accuracy."),
        ("How would you deploy this on an NVIDIA Jetson?",
         f"Start from the verified ONNX file and build a TensorRT engine on the device (for example with trtexec), first in FP16, then INT8 with a calibration set of a few hundred representative training images. Re-run the same verification as for ONNX: top-1 agreement and accuracy on the test split. Then benchmark on the device with warm-up and p95. Run it through TensorRT directly, ONNX Runtime's TensorRT execution provider, or DeepStream for camera pipelines. The CNN might be the better edge choice; it's {c.loc['cnn', 'params_inference'] / 1e6:.1f}M parameters, but accuracy and latency would decide. This is a plan; I haven't run it on a Jetson."),
        ("What would you change with more time?",
         "Multiple seeds per model to measure training variance; the full 101 classes; fine-tuning CLIP's image encoder; a food-specific detector or segmenter trained on labelled boxes, replacing the COCO proposals, for portion-aware nutrition; register-token ViTs for cleaner explanations; CI running the tests on every push; and monitoring in the API (confidence and drift)."),
        ("Why EfficientNet-B0 and ViT-S rather than bigger models?",
         "They train in minutes on a laptop GPU (MPS) and are realistic deployment sizes. The point was a fair comparison between architecture families under identical data and evaluation, not maximum accuracy. The config switches to ViT-B/16 in one line, and the Colab notebook offers it."),
        ("Explain how a Vision Transformer works.",
         "The image is cut into 16×16 patches (196 for 224×224). Each patch is linearly embedded into a token with a position embedding, and a learnable [CLS] token is prepended. Transformer blocks apply multi-head self-attention, so every patch can attend to every other patch from the first layer, plus MLPs with residual connections. The [CLS] token's final state goes to the classification head. It has weaker image-specific inductive biases than a CNN, so pretraining scale matters."),
        ("Explain CLIP and how zero-shot classification works.",
         f"CLIP trains an image encoder and a text encoder contrastively on image–caption pairs: matching pairs are pulled together in a shared embedding space and mismatched pairs pushed apart. For zero-shot classification I turn each class into prompts like \"{F['cfg']['models']['clip']['templates'][0]}\", average the normalised text embeddings over {len(F['cfg']['models']['clip']['templates'])} templates, and predict the class with the highest cosine similarity to the image embedding, scaled by CLIP's learned temperature."),
        ("Why the different learning rates for the CNN and the ViT?",
         f"{F['cfg']['models']['cnn']['lr']} for the CNN and {F['cfg']['models']['vit']['lr']} for the ViT. Transformers are more sensitive to large updates of their pretrained weights, so ViT fine-tuning typically uses a smaller LR. Both use warmup, which protects the pretrained weights while the new head is still random."),
        ("What does warmup + cosine do, and why per step?",
         "Warmup ramps the LR linearly over the first epoch, so the large gradients from the new random head don't damage the pretrained features. Cosine then decays the LR smoothly to 1% of the peak, which helps the model settle. Updating per step rather than per epoch gives a smooth schedule, and a unit test checks its values."),
        ("What is label smoothing and why use it here?",
         f"The target puts about {100 * (1 - F['cfg']['train']['label_smoothing']):.0f}% on the true class and spreads the rest over the others. It reduces over-confidence and is robust to label noise, and Food-101's training labels are known to be noisy."),
        ("How does mixed precision work, and what went wrong?",
         "Autocast runs most operations in fp16 and keeps sensitive ones in fp32. A GradScaler multiplies the loss so small gradients don't underflow. If gradients overflow, that step is skipped and the scale is lowered. That happened on the ViT's first step and caused a harmless 'scheduler stepped before optimizer' warning. I filtered that warning with a comment rather than changing the training logic, so the recorded results stay reproducible."),
        ("How did you choose the checkpoint and avoid overfitting to test?",
         f"Early stopping on val top-1 with patience {F['cfg']['train']['patience']}; the best-val checkpoint is saved and evaluated once on test. The ViT peaked at epoch {t['vit']['best_epoch']} and stopped at {t['vit']['epochs_trained']}. Test never influenced any decision."),
        ("How do you know the differences between models are real?",
         "Exact McNemar tests on paired predictions, using only the images where two models disagree. CNN vs ViT and CNN vs CLIP are significant at p &lt; 0.05; ViT vs CLIP is not. One caveat: that covers test-set sampling, not training randomness, for which I'd need multiple seeds."),
        ("Explain Grad-CAM.",
         "Take the activations of the last convolutional block, backpropagate the predicted class score to get their gradients, average the gradients over space to get one weight per channel, sum the weighted channels, apply ReLU and upsample. In this project it consistently highlights the food itself."),
        ("Your ViT attention maps highlight the background. Is the model broken?",
         f"No. That's the artifact-token effect from 'Vision Transformers Need Registers'. I tested it: over 100 test images, {pct(art['mean_topk_overlap'], 0)} of rollout's top-5 patches are also top-5 by token norm (chance is {pct(art['chance_topk_overlap'], 1)}), and outlier tokens reach {max(a['median_max_over_median_norm'] for a in F['artifact']):.1f}× the median norm. My first test looked after the final LayerNorm, which hides the effect, so I had to measure in the residual stream. Plain rollout is misleading here; register tokens are the documented fix."),
        ("How did you verify the ONNX export?",
         f"Three separate checks on all {ov['conversion']['num_images']} test images: identical tensors into PyTorch and ONNX Runtime (max logit difference {ov['conversion']['max_abs_logit_diff']:.1e}, 100% top-1 agreement); my NumPy preprocessing against torchvision (identical, difference 0.0); and the full serving path's accuracy, which equals PyTorch fp32 at {pct(ov['end_to_end_onnx_top1'])}. The 0.05-point gap to my recorded {pct(ov['recorded_pytorch_top1'])} is fp16 vs fp32, and the report documents it."),
        ("Why was ONNX Runtime faster for one model and slower for the other?",
         f"For EfficientNet, this PyTorch build lacks oneDNN, so depthwise convolutions use a slow fallback: {bms(F, 'cnn', 'PyTorch CPU'):.0f} ms against {bms(F, 'cnn', 'ONNX Runtime CPU'):.1f} ms on ORT. The ViT is matrix-multiply-bound, and on macOS PyTorch most likely benefits from Apple's Accelerate BLAS ({bms(F, 'vit', 'PyTorch CPU'):.1f} vs {bms(F, 'vit', 'ONNX Runtime CPU'):.1f} ms). CoreML only partly supports the ViT graph, so it bounces between CoreML and CPU. Lesson: benchmark on the target."),
        ("What is an ONNX Runtime execution provider?",
         "The backend that actually runs the graph: CPU, CoreML (Apple), CUDA or TensorRT (NVIDIA), and others. ORT assigns each node to the first provider in your list that supports it; unsupported parts fall back to CPU, which creates partitions and copying overhead."),
        ("Why keep PyTorch out of the Docker image?",
         "Smaller image, faster cold starts, a smaller attack surface, and fewer dependency conflicts. The cost is re-implementing preprocessing, so I made it provably identical to torchvision with parity tests and checked it on the full test set."),
        ("How does the API handle bad input?",
         "Content-type allow-list (415), size limit read with a cap (413), empty or undecodable images (400, with a forced full decode and decompression-bomb protection), and top_k validated by FastAPI (422). The model loads once at start-up; the endpoint is a sync function so inference runs in a thread pool."),
        ("How do you test an ML system without the dataset?",
         f"Fixtures build a fake Food-101 tree and a tiny real ONNX model, so the data pipeline and API run against real files and a real ORT session. {F['n_tests']} tests run in a few seconds. I also ran mutation checks: deliberately broke the preprocessing to confirm the tests fail. That exposed a blind spot with banker's rounding, which I fixed with edge-case image sizes."),
        ("Where do the nutrition numbers come from?",
         "They are approximate values for one typical serving, written for this project, in the range of public database entries, and checked for internal consistency (calories ≈ 4P + 4C + 9F) by a test. They are not measurements, and the API returns a disclaimer with every response. Per-photo nutrition needs portion estimation, which is in the v2 roadmap."),
        ("How did you add multi-dish support without any box labels?",
         f"Food-101 has one label per image and no boxes, so I couldn't train or evaluate a food detector. Instead I made it two-stage: a COCO-pretrained YOLO11n, exported to ONNX, proposes food regions, and my already-verified ViT classifies each crop, which is then joined with nutrition. If nothing is detected, it falls back to classifying the whole image, which is exactly the /predict result. I state clearly that there's no mAP, because there's no labelled data to compute it on."),
        ("Which COCO classes do you keep, and why exclude fruit and vegetables?",
         f"Prepared dishes plus bowl: {', '.join(F['food_labels'])}. Bowl matters because curries, ramen and rice come in bowls. I excluded raw produce because the classifier is closed-set: it has no 'carrot' class, so a garnish crop gets forced into a dish. I actually saw that in a demo, where carrot garnish on a steak plate came out as samosa. The list is a constructor argument, so it's a configuration decision, not hard-coded."),
        ("Why re-implement letterbox and NMS instead of using Ultralytics at serving time?",
         "To keep the serving image small and torch-free, the same principle as the classifier: Ultralytics pulls in PyTorch and OpenCV. The ONNX file outputs raw (1, 84, 8400) predictions, so I wrote the letterbox, decode, class-aware NMS and box mapping in about 60 lines of NumPy. The cost is the risk of subtle mismatches, so I verified it against Ultralytics itself and unit-tested every step with a tiny synthetic ONNX detector."),
        ("How does class-aware NMS work in your code?",
         "First I pick each candidate's best class and drop scores below the threshold. Then I add class_id × 7680 to every box coordinate, the same offset Ultralytics uses. Boxes of different classes end up far apart, so a single greedy NMS pass (keep the best box, drop overlaps above the IoU threshold, repeat) only ever suppresses boxes of the same class. A test has a duplicate bowl that must be suppressed and a different-class box that must survive."),
        ("Your detector verification failed at first. What happened?",
         f"Against Ultralytics' PyTorch weights, only {pct(F['yolo']['comparisons']['pytorch_rect']['match_rate_vs_reference'])} / {pct(F['yolo']['comparisons']['pytorch_rect']['match_rate_vs_onnx'])} of detections matched, below my 95% bar. The cause was the reference, not my code: .pt inference is rectangular (for example 640×480), while the static ONNX export always sees 640×640, so the inputs differed. I switched the pass/fail reference to Ultralytics running the same ONNX file, which isolates my pre/post-processing, and got {pct(F['yolo']['comparisons']['same_onnx']['match_rate_vs_reference'])} / {pct(F['yolo']['comparisons']['same_onnx']['match_rate_vs_onnx'])} with mean IoU {F['yolo']['comparisons']['same_onnx']['mean_iou_matched']:.4f}. I kept the PyTorch row as information, because it shows the real cost of a static export."),
        ("What are the weaknesses of the detection pipeline?",
         "There's no measured detection accuracy. COCO's food classes are coarse: a samosa is proposed as a 'sandwich', and dishes outside the six classes are only covered by the whole-image fallback. The classifier can't say 'none of these', so a sauce bowl still gets a dish label. Crops are also somewhat out of distribution for a model trained on whole photos. And the YOLO weights are AGPL-3.0, which constrains licensing; I relicensed the repo as AGPL-3.0 for that reason."),
        ("How would you make the detection production-grade?",
         "Fine-tune a detector or segmenter on box/mask-labelled food data (e.g. UECFOOD-256) and report mAP and per-dish accuracy on a held-out set. Add an open-set 'unknown' option, via a confidence threshold or an extra background class, so garnish and sauces aren't forced into dishes. Use masks plus a size reference for portion estimation. If the AGPL licence is a problem, switch to a permissive detector such as RT-DETR or torchvision's, and re-run the same verification approach."),
        ("Why is the AGPL licence relevant, and what did you do about it?",
         "Ultralytics' code and YOLO weights are AGPL-3.0, and my served detector is derived from those weights. AGPL treats network use as distribution: anyone running a modified version as a service must offer its source to users. I added an AGPL-3.0 LICENSE and a README section explaining why. The detector is also optional in Docker: without it, /predict still works and /detect returns 503. For a commercial product I'd use a permissive detector or buy an Ultralytics licence."),
        ("How would you scale this to all 101 classes or more?",
         "Change the class list in the config; the pipeline is class-agnostic. Expect lower accuracy and more fine-grained confusions, so use more data per class and a bigger backbone on a GPU (the Colab notebook), and possibly fine-tuned CLIP. For new dishes without data, CLIP zero-shot gives a starting point."),
        ("What was the hardest bug?",
         "Several good ones: the download that restarted from zero after a connection reset (fixed with HTTP Range resume); the artifact-token test that showed nothing because I measured after LayerNorm; and the parity test that passed a real bug because of banker's rounding. In each case I verified the cause with evidence before fixing it."),
        ("How is this connected to your published paper?",
         "It rebuilds the recognition and nutrition parts of my final-year paper with a stronger engineering process: reproducible splits, a three-way model comparison with significance testing, explainability, verified export and a containerised API. Meal recording is the next step. (Add your paper's specifics here before the interview: dataset, model and accuracy, so you can contrast old and new.)"),
    ]
    out = ["# 8. Interview preparation {#interview}\n",
           f"{len(qa)} likely questions with model answers. Every number comes from the recorded files. Practise saying the answers out loud and adapt the wording to your own voice.\n"]
    for i, (q, a) in enumerate(qa, 1):
        out.append(f'<div class="qa" markdown="1">\n<p class="q">Q{i}. {html.escape(q)}</p>\n\n{a}\n\n</div>\n')
    script = f"""
## 8.2 Two-minute spoken walkthrough

> "This project is a rebuild of my final-year project and published paper on food recognition with nutritional insights. The goal: take a photo of a meal, recognise the dish, and return approximate nutrition.
>
> I used {len(F['classes'])} classes from Food-101, including samosa and chicken curry, plus some deliberately confusing pairs. The first thing I did was make the data trustworthy. Train and validation come from the official training images and test from the official test images, so there's no leakage. Everything is seeded and versioned, and my EDA hashed every image, which caught a train/val duplicate and one photo filed under two labels.
>
> Then I compared three approaches on identical data. A fine-tuned EfficientNet-B0 reached {pct(t['cnn']['top1'])}, a fine-tuned ViT-S/16 {pct(t['vit']['top1'])}, and zero-shot CLIP, with no training at all, {pct(t['clip']['top1'])}. Rather than eyeballing that, I ran McNemar tests: CLIP and the ViT are statistically tied, and both beat the CNN.
>
> The error analysis showed steak versus filet mignon is the main weakness for every model, and many confident errors are multi-dish photos. For explainability, Grad-CAM showed the CNN focusing on the food, while the ViT's attention rollout lit up the background. I showed that comes from known high-norm artifact tokens rather than the model looking at the table.
>
> For deployment I chose the ViT: same accuracy as CLIP at a quarter of the parameters. I exported it to ONNX and verified 100% top-1 agreement with PyTorch on all {ov['conversion']['num_images']} test images, re-implemented preprocessing without PyTorch and proved it identical, and served it with FastAPI in Docker. Benchmarking showed the fastest runtime depends on the architecture: ONNX Runtime was {bms(F, 'cnn', 'PyTorch CPU') / bms(F, 'cnn', 'ONNX Runtime CPU'):.0f}× faster for the CNN but slower for the ViT on my Mac.
>
> Finally, I added multi-dish detection. Food-101 has no boxes, so a COCO-pretrained YOLO proposes food regions and my ViT classifies each crop. I re-implemented YOLO's pre- and post-processing in NumPy to stay torch-free, and verified it against Ultralytics at {pct(F['yolo']['comparisons']['same_onnx']['match_rate_vs_reference'], 1)} detection agreement. The first verification failed because I'd picked the wrong reference, and that turned into a useful lesson.
>
> It's covered by {F['n_tests']} tests. Next, I'd train a food-specific detector or segmenter for portion-aware nutrition, and use TensorRT for edge devices."
"""
    return "\n".join(out[:2]) + "\n## 8.1 Questions and model answers\n\n" + "\n".join(out[2:]) + script


def sec_glossary() -> str:
    terms = [
        ("AGPL-3.0", "GNU Affero GPL: copyleft licence that also treats offering software over a network as distribution (Ultralytics YOLO uses it)."),
        ("AdamW", "Adam optimiser with weight decay applied directly to the weights (decoupled) rather than through the gradient."),
        ("Artifact tokens", "Background patch tokens a ViT repurposes as global storage; they have very high norms and attract attention (Darcet et al., 2023)."),
        ("Attention rollout", "Explanation method that multiplies head-averaged attention matrices (plus identity for residuals) across layers to estimate each patch's contribution to [CLS]."),
        ("Augmentation", "Random label-preserving changes to training images (crops, flips, colour) that teach invariances and reduce overfitting."),
        ("Autocast / AMP", "Automatic mixed precision: run most operations in 16-bit floats and sensitive ones in 32-bit."),
        ("Banker's rounding", "Rounding .5 to the nearest even number (Python's round): round(0.5)=0, round(1.5)=2."),
        ("Batch size", "Number of images processed together in one optimisation step."),
        ("bf16 / fp16 / fp32", "Floating-point formats: 32-bit (full precision), 16-bit (small range), bfloat16 (16-bit with fp32's range)."),
        ("BLAS", "Basic Linear Algebra Subprograms: optimised matrix-multiply libraries (e.g. Apple Accelerate)."),
        ("[CLS] token", "Extra learnable token in a ViT whose final state is used for classification."),
        ("CLIP", "Contrastive Language-Image Pretraining: paired image and text encoders trained to embed matching pairs close together."),
        ("CNN", "Convolutional neural network: stacks of learned sliding filters that build features from edges to objects."),
        ("Confusion matrix", "Table of true class (rows) vs predicted class (columns)."),
        ("Contrastive learning", "Learning by pulling matching pairs together and pushing non-matching pairs apart in embedding space."),
        ("CoreML", "Apple's on-device ML framework; usable from ONNX Runtime as an execution provider."),
        ("Cosine schedule", "Learning rate that decays following half a cosine curve."),
        ("Depthwise-separable convolution", "Per-channel spatial convolution followed by a 1×1 channel-mixing convolution; much cheaper than standard convolution."),
        ("Docker", "Packages an application and its dependencies into an image that runs identically anywhere."),
        ("Early stopping", "Stop training when the validation metric hasn't improved for a set number of epochs (patience)."),
        ("EDA", "Exploratory data analysis: inspecting data quality and distribution before modelling."),
        ("Embedding", "A vector representation where similar things are close together."),
        ("Epoch", "One full pass over the training data."),
        ("Execution provider (EP)", "The hardware backend ONNX Runtime uses to run (part of) a graph: CPU, CoreML, CUDA, TensorRT…"),
        ("F1 score", "Harmonic mean of precision and recall; macro-F1 averages it over classes."),
        ("FastAPI", "Python web framework that maps functions to HTTP endpoints with automatic validation and docs."),
        ("Fine-grained classification", "Distinguishing visually similar sub-categories (steak vs filet mignon)."),
        ("Fine-tuning", "Continuing to train a pretrained model on a new task."),
        ("Fingerprint (split)", "Hash of all settings that determine the split; a mismatch triggers a rebuild."),
        ("Fixture (pytest)", "Reusable test setup injected into tests by name."),
        ("Food-101", "Public dataset: 101 dishes × 1,000 images; official 750 train / 250 test per class."),
        ("GEMM / MatMul", "General matrix multiplication, the dominant operation in transformers."),
        ("Grad-CAM", "Gradient-weighted class activation mapping: heatmap from gradient-weighted conv feature maps."),
        ("GradScaler", "Scales the loss in fp16 training so small gradients don't underflow; skips steps that overflow."),
        ("Head", "The final layer(s) mapping features to class scores; replaced when fine-tuning for new classes."),
        ("Inductive bias", "Assumptions built into an architecture (e.g. CNN locality) that help it learn from less data."),
        ("INT8 quantisation", "Running a model with 8-bit integers for speed and memory, calibrated on representative data."),
        ("IoU", "Intersection over union: overlap area of two boxes divided by the area of their union (0 to 1)."),
        ("Label smoothing", "Softening one-hot targets (e.g. 0.9 on the true class) to reduce over-confidence."),
        ("LayerNorm", "Normalises each token's features; in ViTs it hides token-norm differences."),
        ("Leakage", "Information from val/test reaching training, which inflates scores."),
        ("Letterbox", "Resize keeping aspect ratio, then pad to a fixed square input (YOLO: 640×640, grey 114)."),
        ("Logits", "Raw class scores before softmax."),
        ("mAP", "Mean average precision: the standard detection metric, averaging precision over recall levels, classes and IoU thresholds."),
        ("Macro average", "Average of a per-class metric where every class counts equally."),
        ("McNemar's test", "Paired significance test using only the samples where two classifiers disagree."),
        ("MD5", "Hash function; identical files have identical hashes (used for duplicate detection)."),
        ("MPS", "Metal Performance Shaders: PyTorch's GPU backend on Apple Silicon."),
        ("Mutation testing", "Deliberately introducing bugs to check that tests catch them."),
        ("NMS", "Non-maximum suppression: keep the best of overlapping duplicate boxes, drop the rest above an IoU threshold."),
        ("Normalisation (input)", "Subtracting the dataset mean and dividing by std per channel."),
        ("ONNX", "Open Neural Network Exchange: portable file format for model graphs and weights."),
        ("ONNX Runtime (ORT)", "Inference engine that optimises and runs ONNX models on many backends."),
        ("oneDNN", "Intel's optimised deep-learning CPU kernels; missing from the PyTorch build used here."),
        ("Open-set / closed-set", "A closed-set classifier must pick one of its known classes; an open-set one can answer 'unknown'."),
        ("Opset", "Version of the ONNX operator set a model uses."),
        ("Overfitting", "Fitting training data much better than unseen data (large train/val gap)."),
        ("p95 latency", "95th percentile: 95% of requests were at least this fast."),
        ("Patch (ViT)", "A 16×16-pixel image tile turned into one token."),
        ("Precision / recall", "Of predicted X, how many were X / of real X, how many were found."),
        ("Prompt ensembling", "Averaging text embeddings from several prompt templates per class."),
        ("Pydantic", "Data validation library used by FastAPI for request/response schemas."),
        ("Rectangular inference", "Ultralytics pads images only to a stride multiple (e.g. 640×480) instead of a full square; faster, but differs from a static export."),
        ("Register tokens", "Extra learnable ViT tokens that absorb global computation, removing artifact tokens."),
        ("Residual connection", "Adding a layer's input to its output, letting information and gradients flow past the layer."),
        ("Self-attention", "Each token weights every other token by query–key similarity and mixes their values."),
        ("Softmax", "Turns logits into probabilities that sum to 1."),
        ("TensorRT", "NVIDIA's optimising inference compiler/runtime for its GPUs and Jetson devices."),
        ("Top-k accuracy", "Share of samples whose true class is among the k highest-scoring predictions."),
        ("Two-stage detection (here)", "Detector proposes regions, a separate classifier labels each crop."),
        ("Transfer learning", "Reusing a model trained on a large dataset as the starting point for a new task."),
        ("Uvicorn", "ASGI web server that runs the FastAPI app."),
        ("ViT", "Vision Transformer: transformer encoder over image patches."),
        ("Warmup", "Gradually increasing the learning rate at the start of training."),
        ("Weight decay", "Regularisation that shrinks weights towards zero."),
        ("YOLO", "'You Only Look Once': family of fast single-stage object detection/segmentation models."),
        ("Zero-shot", "Performing a task without task-specific training examples (here: classifying via text prompts)."),
    ]
    terms.sort(key=lambda t: re.sub(r"[^a-z0-9]", "", t[0].lower()))
    return "# 9. Glossary {#glossary}\n\n" + md_table(["Term", "Meaning"], [[f"**{a}**", b] for a, b in terms])


# =========================================================================== assembly


CSS = r"""
@page { size: A4; margin: 18mm 16mm 20mm 16mm;
  @top-right { content: "food-vision · study guide"; font-size: 8pt; color: #888; }
  @bottom-center { content: counter(page); font-size: 9pt; color: #555; } }
@page :first { @top-right { content: none; } @bottom-center { content: none; } }
html { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.42; color: #1d1d1f; }
h1 { font-size: 20pt; color: #1F3B5C; border-bottom: 2px solid #1F3B5C; padding-bottom: 3pt; margin-top: 0;
     break-before: page; bookmark-level: 1; }
h2 { font-size: 14pt; color: #2B5383; margin-top: 16pt; bookmark-level: 2; break-after: avoid; }
h3 { font-size: 11.5pt; color: #333; margin-top: 12pt; bookmark-level: 3; break-after: avoid; }
p, li { orphans: 2; widows: 2; }
code { font-family: Menlo, "SF Mono", monospace; font-size: 8.4pt; background: #F2F4F7; padding: 0 2pt; border-radius: 2pt; }
pre, .codehilite { font-family: Menlo, monospace; font-size: 7.5pt; line-height: 1.35; background: #F6F8FA;
       border: 1px solid #E1E4E8; border-radius: 4pt; padding: 6pt 8pt; white-space: pre-wrap; word-wrap: break-word; }
.codehilite pre { border: none; padding: 0; margin: 0; background: none; }
pre code, .codehilite code { background: none; padding: 0; font-size: 7.5pt; }
.codetitle { font-size: 8pt; color: #666; margin-bottom: -6pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 8.6pt; break-inside: auto; }
th { background: #1F3B5C; color: white; text-align: left; padding: 4pt 5pt; }
td { border-bottom: 1px solid #DDD; padding: 3.5pt 5pt; vertical-align: top; }
tr { break-inside: avoid; }
tr:nth-child(even) td { background: #F7F9FC; }
figure { margin: 8pt 0 12pt; text-align: center; break-inside: avoid; }
figure img { max-height: 225mm; object-fit: contain; }
figcaption { font-size: 8.4pt; color: #555; margin-top: 3pt; font-style: italic; }
.box { border-left: 4pt solid #4C72B0; background: #EEF3FA; padding: 6pt 10pt; margin: 9pt 0; border-radius: 3pt; break-inside: avoid; }
.box p, .box li { font-size: 9.4pt; }
.box.why { border-color: #55A868; background: #EEF7F0; }
.box.caveat { border-color: #C44E52; background: #FBEFEF; }
.box.interview { border-color: #8172B3; background: #F3F0FA; }
.boxtitle { font-weight: bold; margin: 0 0 3pt; color: #1F3B5C; }
.qa { border: 1px solid #E3E3E8; border-radius: 4pt; padding: 5pt 9pt; margin: 7pt 0; break-inside: avoid; }
.q { font-weight: bold; color: #2B5383; margin: 0 0 3pt; }
blockquote { border-left: 3pt solid #BBB; margin: 6pt 0; padding: 2pt 10pt; color: #333; font-style: italic; }
.small { font-size: 8.3pt; color: #555; }
.title-page { text-align: center; padding-top: 55mm; }
.title-page h1.t { font-size: 30pt; border: none; break-before: avoid; color: #1F3B5C; margin-bottom: 4pt; bookmark-level: none; }
.title-page .sub { font-size: 14pt; color: #444; }
.title-page .meta { margin-top: 30mm; font-size: 10pt; color: #666; line-height: 1.7; }
.toc { break-before: page; }
.toc h1 { break-before: avoid; }
.toc ul { list-style: none; padding-left: 0; }
.toc li { margin: 2.5pt 0; }
.toc li.l2 { padding-left: 14pt; font-size: 9.4pt; }
.toc a { color: #1d1d1f; text-decoration: none; }
.toc a::after { content: leader('.') target-counter(attr(href), page); }
.missing { color: #C44E52; }
.keep { break-inside: avoid; }
"""


def build_html(F: dict, figs: dict) -> str:
    body_md = "\n".join([
        sec_overview(F), sec_architecture(F, figs), sec_phase1(F, figs), sec_phase2(F, figs),
        sec_phase3(F, figs), sec_phase4(F, figs), sec_phase5(F), sec_detection(F, figs), sec_deep_dive(F), sec_code_tour(F),
        sec_how_to_run(F), sec_limitations().replace("{{", "{").replace("}}", "}"), sec_interview(F), sec_glossary(),
    ])
    md = markdown.Markdown(extensions=["tables", "fenced_code", "codehilite", "attr_list", "md_in_html",
                                       "sane_lists", "toc"],
                           extension_configs={"codehilite": {"guess_lang": False, "css_class": "codehilite"}})
    body = md.convert(body_md)

    toc_items = []
    for level, hid, text in re.findall(r'<h([12]) id="([^"]+)">(.*?)</h\1>', body):
        clean = re.sub(r"<[^>]+>", "", text)
        toc_items.append(f'<li class="l{level}"><a href="#{hid}">{clean}</a></li>')
    toc = f'<div class="toc"><h1 style="bookmark-level:none">Contents</h1><ul>{"".join(toc_items)}</ul></div>'

    title = f"""<div class="title-page">
<h1 class="t">food-vision</h1>
<p class="sub">Study guide: food recognition, nutritional insights and deployment<br>
CNN vs ViT vs zero-shot CLIP on Food-101</p>
<p class="meta">Personal interview-preparation notes<br>
Built {date.today().isoformat()} from the repository's recorded outputs<br>
(configs/, runs/, reports/). Every number is read from those files.</p></div>"""
    pyg = HtmlFormatter(style="friendly").get_style_defs(".codehilite")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>food-vision study guide</title>"
            f"<style>{CSS}\n{pyg}</style></head><body>{title}{toc}{body}</body></html>")


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    exe = shutil.which("weasyprint")
    if exe:
        subprocess.run([exe, str(html_path), str(pdf_path)], check=True)
        return
    from weasyprint import HTML  # pip install weasyprint

    HTML(filename=str(html_path)).write_pdf(str(pdf_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="docs/food-vision-guide.pdf")
    parser.add_argument("--keep-build", help="directory to keep the intermediate HTML and figures")
    args = parser.parse_args()

    F = load_facts()
    with tempfile.TemporaryDirectory() as tmp:
        build = Path(args.keep_build) if args.keep_build else Path(tmp)
        build.mkdir(parents=True, exist_ok=True)
        figs = make_figures(F, build)
        html_path = build / "guide.html"
        html_path.write_text(build_html(F, figs))
        out = ROOT / args.out
        render_pdf(html_path, out)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
