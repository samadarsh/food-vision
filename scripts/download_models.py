"""Download the exported ONNX models from the GitHub release into models/.

Lets you run the API, the CLI and the Docker build without downloading Food-101 or training:
    python scripts/download_models.py            # vit.onnx + yolo11n.onnx
    python scripts/download_models.py --tag v0.1.0 --only vit.onnx

Uses only the standard library. Existing files are kept unless --force is given.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "samadarsh/food-vision"
DEFAULT_TAG = "v0.1.0"
FILES = ("vit.onnx", "yolo11n.onnx")  # classifier (required by the API), detector (enables /detect)


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest` via a temporary file, so an interrupted download leaves nothing behind."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default=DEFAULT_TAG, help=f"release tag (default {DEFAULT_TAG})")
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--only", choices=FILES, action="append", help="download just this file (repeatable)")
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    failed = False
    for name in args.only or FILES:
        dest = args.out / name
        if dest.exists() and not args.force:
            print(f"{dest} exists, skipping (use --force to replace)")
            continue
        url = f"https://github.com/{REPO}/releases/download/{args.tag}/{name}"
        print(f"{url} -> {dest}")
        try:
            download(url, dest)
        except urllib.error.HTTPError as e:
            failed = True
            print(f"  failed: HTTP {e.code}. Is {name} attached to release {args.tag}?", file=sys.stderr)
        else:
            print(f"  {dest.stat().st_size / 1e6:.1f} MB")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
