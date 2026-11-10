"""Exploratory data analysis of the configured Food-101 subset.

Outputs (in <reports_dir>/eda/):
    class_counts.csv / .png   images per class per split
    image_stats.csv           per-image width, height, mode, bytes, md5, decode status
    image_sizes.png           width vs height scatter + longest-side / aspect histograms
    sample_grid.jpg           a few random images per class
    corrupt_images.txt        images that fail to decode (empty if none)
    duplicates.csv            byte-identical files (flags any cross-split leakage)
    summary.json              headline numbers

Usage:
    python scripts/eda.py [--config configs/default.yaml] [--samples-per-class 6]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from food_vision.config import DEFAULT_CONFIG, load_config  # noqa: E402
from food_vision.data import SPLITS, Sample, food101_dir, get_splits  # noqa: E402

SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}


def inspect_image(path: Path) -> dict:
    """Header info + full decode check + md5 for one image."""
    rec: dict = {"width": None, "height": None, "mode": None, "ok": True, "error": ""}
    try:
        data = path.read_bytes()
        rec["bytes"] = len(data)
        rec["md5"] = hashlib.md5(data).hexdigest()
        with Image.open(path) as im:
            im.verify()  # structural check
        with Image.open(path) as im:  # verify() invalidates the handle; reopen to decode
            rec["width"], rec["height"], rec["mode"] = im.width, im.height, im.mode
            im.load()  # full decode catches truncated files verify() misses
    except Exception as e:  # noqa: BLE001 - we want to record any failure
        rec["ok"], rec["error"] = False, f"{type(e).__name__}: {e}"
    return rec


def collect_stats(splits: dict[str, list[Sample]], images_dir: Path, workers: int) -> pd.DataFrame:
    """Inspect every image in every split (threaded; I/O + decode bound)."""
    rows = [(s, x) for s in SPLITS for x in splits[s]]

    def work(item: tuple[str, Sample]) -> dict:
        split, x = item
        return {"split": split, "class_name": x.class_name, "rel_path": x.rel_path,
                **inspect_image(images_dir / x.rel_path)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(tqdm(pool.map(work, rows), total=len(rows), desc="Inspecting images"))
    return pd.DataFrame(records)


def plot_class_counts(counts: pd.DataFrame, out: Path) -> None:
    """Grouped horizontal bar chart of images per class per split."""
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(counts) + 1.5))
    counts[list(SPLITS)].plot.barh(ax=ax, color=[SPLIT_COLORS[s] for s in SPLITS], width=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("images")
    ax.set_ylabel("")
    ax.set_title("Images per class per split")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_image_sizes(df: pd.DataFrame, out: Path) -> None:
    """Width/height scatter and longest-side + aspect-ratio histograms."""
    ok = df[df.ok]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(ok.width, ok.height, s=4, alpha=0.3, color="#4C72B0")
    axes[0].set(xlabel="width (px)", ylabel="height (px)", title="Image dimensions")
    axes[1].hist(ok[["width", "height"]].max(axis=1), bins=40, color="#4C72B0")
    axes[1].set(xlabel="longest side (px)", ylabel="images", title="Longest side")
    axes[2].hist(ok.width / ok.height, bins=40, color="#4C72B0")
    axes[2].set(xlabel="aspect ratio (w / h)", ylabel="images", title="Aspect ratio")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_sample_grid(splits: dict[str, list[Sample]], images_dir: Path, class_names: list[str],
                     per_class: int, seed: int, out: Path, skip: set[str]) -> None:
    """Rows = classes, columns = random training images (centre-cropped to square).
    Paths in `skip` (corrupt images) are never drawn."""
    rng = random.Random(seed)
    by_class: dict[str, list[Sample]] = {c: [] for c in class_names}
    for x in splits["train"]:
        if x.rel_path not in skip:
            by_class[x.class_name].append(x)

    fig, axes = plt.subplots(len(class_names), per_class,
                             figsize=(per_class * 1.6, len(class_names) * 1.6))
    for r, name in enumerate(class_names):
        picks = rng.sample(by_class[name], min(per_class, len(by_class[name])))
        for c in range(per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(picks):
                with Image.open(images_dir / picks[c].rel_path) as im:
                    im = im.convert("RGB")
                    side = min(im.size)
                    left, top = (im.width - side) // 2, (im.height - side) // 2
                    ax.imshow(im.crop((left, top, left + side, top + side)).resize((160, 160)))
        axes[r, 0].text(-0.08, 0.5, name, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)
    fig.subplots_adjust(left=0.16, right=0.99, top=0.99, bottom=0.01, wspace=0.04, hspace=0.04)
    fig.savefig(out, dpi=90, pil_kwargs={"quality": 80, "optimize": True})  # JPEG: photos compress far better
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--samples-per-class", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = get_splits(cfg)
    images_dir = food101_dir(cfg.data.root) / "images"
    out = cfg.data.reports_dir / "eda"
    out.mkdir(parents=True, exist_ok=True)
    names = cfg.data.class_names

    # 1. Class counts
    counts = pd.DataFrame(
        {s: pd.Series([x.class_name for x in splits[s]]).value_counts() for s in SPLITS}
    ).reindex(names).fillna(0).astype(int)
    counts.index.name = "class_name"
    counts.to_csv(out / "class_counts.csv")
    plot_class_counts(counts, out / "class_counts.png")

    # 2. Per-image stats, corrupt check, duplicates
    df = collect_stats(splits, images_dir, args.workers)
    df.to_csv(out / "image_stats.csv", index=False)
    bad = df[~df.ok]
    (out / "corrupt_images.txt").write_text(
        "".join(f"{r.split}\t{r.rel_path}\t{r.error}\n" for r in bad.itertuples())
    )
    dup = df[df.ok & df.md5.duplicated(keep=False)].sort_values("md5")
    dup[["md5", "split", "class_name", "rel_path"]].to_csv(out / "duplicates.csv", index=False)
    cross_split = int((dup.groupby("md5").split.nunique() > 1).sum()) if len(dup) else 0

    # 3. Plots
    plot_image_sizes(df, out / "image_sizes.png")
    plot_sample_grid(splits, images_dir, names, args.samples_per_class, cfg.seed,
                     out / "sample_grid.jpg", skip=set(bad.rel_path))

    # 4. Summary
    ok = df[df.ok]
    longest = ok[["width", "height"]].max(axis=1)
    summary = {
        "num_classes": len(names),
        "images_per_split": {s: len(splits[s]) for s in SPLITS},
        "corrupt_images": int(len(bad)),
        "non_rgb_images": {str(k): int(v) for k, v in ok[ok["mode"] != "RGB"]["mode"].value_counts().items()},
        "width": ok.width.describe().round(1).to_dict(),
        "height": ok.height.describe().round(1).to_dict(),
        "pct_longest_side_eq_512": round(100 * float((longest == 512).mean()), 2),
        "pct_shorter_side_lt_224": round(100 * float((ok[["width", "height"]].min(axis=1) < 224).mean()), 2),
        "duplicate_groups": int(dup.md5.nunique()) if len(dup) else 0,
        "cross_split_duplicate_groups": cross_split,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nEDA outputs written to {out}/")


if __name__ == "__main__":
    main()
