"""Food-101 download, class-subset sampling, split manifests, Dataset and transforms.

Split strategy
--------------
Food-101 ships an official split of 750 train / 250 test images per class.
* train + val are sampled *disjointly* from the official train list;
* test is sampled from the official test list.
So no test image is ever seen in training and results stay comparable to the
official protocol. Sampling is deterministic given (seed, classes, counts),
and the resulting split is written to CSV manifests for inspection/reuse.
"""

from __future__ import annotations

import csv
import hashlib
import http.client
import json
import random
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from food_vision.config import Config, DataConfig
from food_vision.utils import seed_worker

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLITS = ("train", "val", "test")
FOOD101_URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
FOOD101_MD5 = "85eeb15f3717b99a5da872d97d918f87"  # same checksum torchvision uses


@dataclass(frozen=True)
class Sample:
    """One image in a split: path relative to the Food-101 images dir, and its label."""

    rel_path: str  # e.g. "samosa/1008131.jpg"
    label: int
    class_name: str


# --------------------------------------------------------------------------- download


def food101_dir(root: Path) -> Path:
    """Directory torchvision extracts Food-101 into."""
    return Path(root) / "food-101"


def _md5(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f, tqdm(total=path.stat().st_size, unit="B", unit_scale=True,
                                     desc="Verifying MD5") as bar:
        while block := f.read(chunk):
            h.update(block)
            bar.update(len(block))
    return h.hexdigest()


def download_resumable(url: str, dest: Path, max_retries: int = 30, backoff: float = 5.0,
                       chunk: int = 1 << 20, timeout: float = 60.0) -> None:
    """Download `url` to `dest`, resuming from any partial file via HTTP Range requests.

    Survives dropped connections: each failure waits (exponential backoff, capped at
    2 min) and continues from the bytes already on disk. The retry budget resets
    whenever progress is made, so only a server that is truly unreachable gives up.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "food-vision-downloader"}
    with urlopen(Request(url, method="HEAD", headers=headers), timeout=timeout) as resp:
        total = int(resp.headers["Content-Length"])

    failures = 0
    while (have := dest.stat().st_size if dest.exists() else 0) < total:
        req = Request(url, headers={**headers, "Range": f"bytes={have}-"})
        try:
            with urlopen(req, timeout=timeout) as resp:
                if resp.status == 200 and have:  # server ignored Range: start over
                    have = 0
                mode = "ab" if have else "wb"
                with open(dest, mode) as f, tqdm(total=total, initial=have, unit="B",
                                                 unit_scale=True, desc=dest.name) as bar:
                    while block := resp.read(chunk):
                        f.write(block)
                        bar.update(len(block))
                        failures = 0
            if dest.stat().st_size < total and dest.stat().st_size == have:
                raise ConnectionError("server closed the connection without sending data")
        except (OSError, http.client.HTTPException) as e:  # URLError, resets, timeouts
            failures += 1
            if failures > max_retries:
                raise RuntimeError(f"Download failed after {max_retries} retries: {e}") from e
            wait = min(120.0, backoff * 2 ** (failures - 1))
            got = dest.stat().st_size if dest.exists() else 0
            print(f"\n{type(e).__name__}: {e}. Have {got / 1e9:.2f}/{total / 1e9:.2f} GB; "
                  f"retry {failures}/{max_retries} in {wait:.0f}s...")
            time.sleep(wait)
    if dest.stat().st_size > total:
        raise RuntimeError(f"{dest} is larger than expected ({dest.stat().st_size} > {total}); delete it and retry")


def ensure_food101(root: Path) -> Path:
    """Make sure Food-101 is downloaded and extracted under `root`; return its directory.

    Uses a resumable downloader (the ETH server often resets long downloads and
    torchvision's downloader restarts from zero). A marker file records a completed
    extraction, so an interrupted extract is redone rather than silently used.
    """
    root = Path(root)
    base = food101_dir(root)
    marker = base / ".extracted"
    archive = root / "food-101.tar.gz"
    if marker.exists():
        return base
    if not archive.exists() and (base / "meta" / "train.json").exists():
        marker.touch()  # pre-existing extraction (e.g. by torchvision on Colab)
        return base

    print(f"Downloading Food-101 to {archive} (~5 GB, resumable; re-run to continue if stopped)")
    download_resumable(FOOD101_URL, archive)
    if (digest := _md5(archive)) != FOOD101_MD5:
        raise RuntimeError(f"MD5 mismatch for {archive} ({digest} != {FOOD101_MD5}). "
                           "The file is corrupt: delete it and run again.")

    print(f"Extracting to {root} (a few minutes)...")
    with tarfile.open(archive, "r:gz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(root, filter="data")  # blocks path traversal
        else:  # Python < 3.10.12
            tar.extractall(root)
    marker.touch()
    print(f"Done. You can delete {archive} to free ~5 GB.")
    return base


def available_classes(base: Path) -> list[str]:
    """All 101 class names from meta/classes.txt."""
    return (base / "meta" / "classes.txt").read_text().split()


def read_official_split(base: Path, split: str) -> dict[str, list[str]]:
    """Official split as {class: [image ids like 'samosa/1008131', ...]}."""
    with open(base / "meta" / f"{split}.json") as f:
        return json.load(f)


# --------------------------------------------------------------------------- splitting


def _take(ids: list[str], n: int | None) -> list[str]:
    if n is not None and n > len(ids):
        raise ValueError(f"Requested {n} images but only {len(ids)} available")
    return ids if n is None else ids[:n]


def build_splits(cfg: DataConfig, seed: int, base: Path) -> dict[str, list[Sample]]:
    """Deterministically sample the configured class subset into train/val/test."""
    valid = set(available_classes(base))
    unknown = [c for c in cfg.class_names if c not in valid]
    if unknown:
        raise ValueError(f"Not Food-101 classes: {unknown}. See {base / 'meta/classes.txt'}")

    official_train = read_official_split(base, "train")
    official_test = read_official_split(base, "test")
    excluded = set(cfg.exclude)
    splits: dict[str, list[Sample]] = {s: [] for s in SPLITS}

    for label, name in enumerate(cfg.class_names):
        # Per-class RNG keyed on class name: adding/removing a class does not
        # reshuffle the images chosen for the other classes.
        rng = random.Random(f"{seed}-{name}")

        train_pool = sorted(official_train[name])
        rng.shuffle(train_pool)
        train_pool = [i for i in train_pool if i not in excluded]  # filter after shuffle: stable
        n_val = cfg.val_per_class if cfg.val_per_class is not None else len(train_pool) // 10
        val_ids = _take(train_pool, n_val)
        train_ids = _take(train_pool[n_val:], cfg.train_per_class)

        test_pool = sorted(official_test[name])
        rng.shuffle(test_pool)
        test_pool = [i for i in test_pool if i not in excluded]
        test_ids = _take(test_pool, cfg.test_per_class)

        for split, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
            splits[split].extend(Sample(f"{i}.jpg", label, name) for i in ids)

    return splits


def save_splits(splits: dict[str, list[Sample]], out_dir: Path, fingerprint: str) -> None:
    """Write one CSV per split plus a fingerprint file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, samples in splits.items():
        with open(out_dir / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rel_path", "label", "class_name"])
            w.writerows((s.rel_path, s.label, s.class_name) for s in samples)
    (out_dir / "fingerprint.txt").write_text(fingerprint + "\n")


def load_split_csv(path: Path) -> list[Sample]:
    """Read a split manifest written by `save_splits`."""
    with open(path, newline="") as f:
        return [Sample(r["rel_path"], int(r["label"]), r["class_name"]) for r in csv.DictReader(f)]


def get_splits(cfg: Config, rebuild: bool = False) -> dict[str, list[Sample]]:
    """Return the train/val/test manifests, (re)building them if the config changed."""
    d = cfg.data
    fp = d.split_fingerprint(cfg.seed)
    fp_file = d.splits_dir / "fingerprint.txt"
    up_to_date = fp_file.exists() and fp_file.read_text().strip() == fp
    if up_to_date and not rebuild and all((d.splits_dir / f"{s}.csv").exists() for s in SPLITS):
        return {s: load_split_csv(d.splits_dir / f"{s}.csv") for s in SPLITS}

    base = ensure_food101(d.root)
    splits = build_splits(d, cfg.seed, base)
    save_splits(splits, d.splits_dir, fp)
    return splits


# --------------------------------------------------------------------------- dataset


class FoodDataset(Dataset):
    """Images listed in a split manifest; returns (tensor, label)."""

    def __init__(self, samples: list[Sample], images_dir: Path, transform: T.Compose | None = None):
        self.samples = samples
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def path(self, idx: int) -> Path:
        """Absolute path of sample `idx` (used for error analysis)."""
        return self.images_dir / self.samples[idx].rel_path

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | Image.Image, int]:
        s = self.samples[idx]
        with Image.open(self.images_dir / s.rel_path) as im:
            img = im.convert("RGB")  # Food-101 contains a few greyscale / CMYK JPEGs
        if self.transform is not None:
            img = self.transform(img)
        return img, s.label


def build_transforms(
    image_size: int = 224,
    train: bool = False,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> T.Compose:
    """Training augmentation or deterministic eval preprocessing.

    `mean`/`std` default to ImageNet; pass the model's own values (e.g. timm ViTs use 0.5).
    """
    normalize = T.Normalize(mean, std)
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                T.ToTensor(),
                normalize,
            ]
        )
    resize = int(round(image_size / 0.875))  # 224 -> 256, the standard eval ratio
    return T.Compose([T.Resize(resize), T.CenterCrop(image_size), T.ToTensor(), normalize])


def build_datasets(
    cfg: Config,
    train_transform: T.Compose | None = None,
    eval_transform: T.Compose | None = None,
) -> dict[str, FoodDataset]:
    """Datasets for all splits. Transforms default to `build_transforms`; models
    with their own preprocessing (e.g. CLIP) can pass custom ones."""
    splits = get_splits(cfg)
    images_dir = food101_dir(cfg.data.root) / "images"
    size = cfg.data.image_size
    train_tf = train_transform or build_transforms(size, train=True)
    eval_tf = eval_transform or build_transforms(size, train=False)
    return {
        s: FoodDataset(splits[s], images_dir, train_tf if s == "train" else eval_tf)
        for s in SPLITS
    }


def build_dataloaders(
    datasets: dict[str, FoodDataset],
    cfg: Config,
    device: torch.device,
    generator: torch.Generator,
    batch_size: int | None = None,
) -> dict[str, DataLoader]:
    """DataLoaders; only train is shuffled. pin_memory only helps on CUDA."""
    d = cfg.data
    return {
        s: DataLoader(
            ds,
            batch_size=batch_size or d.batch_size,
            shuffle=(s == "train"),
            num_workers=d.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=d.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        for s, ds in datasets.items()
    }
