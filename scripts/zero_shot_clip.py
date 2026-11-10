"""Zero-shot CLIP on the same test split (no training), evaluated like the trained models.

Writes to <runs_dir>/clip/: test_metrics.json, test_predictions.npz, config.json.
Also reports val accuracy, since nothing was tuned on it.

Usage:
    python scripts/zero_shot_clip.py [--config configs/default.yaml] [--limit-batches 3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from food_vision.clip_zero_shot import ZeroShotCLIP
from food_vision.config import DEFAULT_CONFIG, load_config
from food_vision.data import build_dataloaders, build_datasets
from food_vision.device import get_device
from food_vision.engine import evaluate, now, predict
from food_vision.metrics import save_test_outputs
from food_vision.models import count_parameters
from food_vision.utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--limit-batches", type=int, help="cap batches per split (smoke tests)")
    parser.add_argument("--runs-dir", help="override train.runs_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ccfg = cfg.clip
    run_dir = Path(args.runs_dir or cfg.train.runs_dir) / "clip"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**cfg.to_dict(), "model_key": "clip"}, indent=2))

    generator = seed_everything(cfg.seed)
    device = get_device(cfg.train.device)
    class_names = cfg.data.class_names
    t0 = now()
    model = ZeroShotCLIP(ccfg.arch, ccfg.pretrained, class_names, ccfg.templates, device)
    n_params = count_parameters(model.model)
    print(f"CLIP {ccfg.arch} ({ccfg.pretrained}, {n_params / 1e6:.1f}M params) on {device}, "
          f"{len(ccfg.templates)} prompt templates")

    # CLIP must see its own preprocessing; train split is unused (zero-shot).
    datasets = build_datasets(cfg, train_transform=model.preprocess, eval_transform=model.preprocess)
    loaders = build_dataloaders(datasets, cfg, device, generator, batch_size=ccfg.batch_size)

    val = evaluate(model, loaders["val"], device, limit_batches=args.limit_batches)
    logits, labels = predict(model, loaders["test"], device, limit_batches=args.limit_batches, desc="test")
    metrics = save_test_outputs(run_dir, logits, labels, class_names, extra={
        "model": "clip", "arch": f"{ccfg.arch} ({ccfg.pretrained})", "params": n_params,
        "zero_shot": True, "templates": ccfg.templates, "val_top1": val["top1"],
        "val_top5": val["top5"], "minutes": round((now() - t0) / 60, 1), "device": str(device),
    })
    print(f"Val  top-1 {val['top1']:.4f}, top-5 {val['top5']:.4f}")
    print(f"Test top-1 {metrics['top1']:.4f}, top-5 {metrics['top5']:.4f}")
    print(f"Outputs in {run_dir}/")


if __name__ == "__main__":
    main()
