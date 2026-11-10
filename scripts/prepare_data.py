"""Download Food-101 (if needed), build the class-subset split and print a summary.

Usage:
    python scripts/prepare_data.py [--config configs/default.yaml] [--rebuild]
"""

from __future__ import annotations

import argparse
from collections import Counter

from food_vision.config import DEFAULT_CONFIG, load_config
from food_vision.data import SPLITS, get_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--rebuild", action="store_true", help="rebuild manifests even if up to date")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = get_splits(cfg, rebuild=args.rebuild)

    print(f"\nClasses ({len(cfg.data.class_names)}), seed={cfg.seed}")
    counts = {s: Counter(x.class_name for x in splits[s]) for s in SPLITS}
    print(f"{'class':<20}" + "".join(f"{s:>8}" for s in SPLITS))
    for name in cfg.data.class_names:
        print(f"{name:<20}" + "".join(f"{counts[s][name]:>8}" for s in SPLITS))
    print(f"{'TOTAL':<20}" + "".join(f"{len(splits[s]):>8}" for s in SPLITS))

    # Sanity check: the three splits must not share any image.
    sets = {s: {x.rel_path for x in splits[s]} for s in SPLITS}
    overlaps = {
        f"{a}/{b}": len(sets[a] & sets[b])
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1 :]
    }
    assert not any(overlaps.values()), f"Split overlap detected: {overlaps}"
    print(f"\nNo overlap between splits. Manifests in {cfg.data.splits_dir}/")


if __name__ == "__main__":
    main()
