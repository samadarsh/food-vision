"""Fine-tune a classifier (CNN or ViT) on the Food-101 subset, then evaluate it on test.

Writes to <runs_dir>/<model>/:
    metrics.csv            one row per epoch (train/val loss, top-1, top-5, lr, time)
    best.pt                best-val-top1 checkpoint (weights + arch + classes + preprocessing)
    test_metrics.json      top-1 / top-5 of the best checkpoint on the test split
    test_predictions.npz   raw test logits + labels (used by Phase 3)
    config.json            the exact config used

Usage:
    python scripts/train.py --model cnn
    python scripts/train.py --model vit --epochs 10
    python scripts/train.py --model cnn --epochs 1 --limit-batches 5   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch import nn

from food_vision.config import DEFAULT_CONFIG, load_config
from food_vision.data import build_dataloaders, build_datasets, build_transforms
from food_vision.device import get_device
from food_vision.engine import (
    CSVLogger, EarlyStopping, amp_setup, build_optimizer, evaluate, fmt_time, now, predict,
    train_one_epoch, warmup_cosine,
)
from food_vision.metrics import save_test_outputs
from food_vision.models import build_classifier, count_parameters, load_classifier, save_checkpoint
from food_vision.utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="key under `models:` in the config, e.g. cnn or vit")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, help="override train.epochs")
    parser.add_argument("--limit-batches", type=int, help="cap batches per split (smoke tests)")
    parser.add_argument("--runs-dir", help="override train.runs_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model not in cfg.classifiers:
        parser.error(f"--model must be one of {sorted(cfg.classifiers)} (clip has its own script)")
    mcfg, tcfg = cfg.classifiers[args.model], cfg.train
    epochs = args.epochs or tcfg.epochs
    run_dir = Path(args.runs_dir or tcfg.runs_dir) / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**cfg.to_dict(), "model_key": args.model,
                                                      "epochs_run_with": epochs}, indent=2))

    generator = seed_everything(cfg.seed)
    device = get_device(tcfg.device)
    class_names = cfg.data.class_names
    model, pre = build_classifier(mcfg.arch, len(class_names), pretrained=True)
    model.to(device)
    n_params = count_parameters(model)

    datasets = build_datasets(
        cfg,
        train_transform=build_transforms(pre.image_size, True, pre.mean, pre.std),
        eval_transform=build_transforms(pre.image_size, False, pre.mean, pre.std),
    )
    loaders = build_dataloaders(datasets, cfg, device, generator, batch_size=mcfg.batch_size)

    steps_per_epoch = min(len(loaders["train"]), args.limit_batches or len(loaders["train"]))
    optimizer = build_optimizer(model, mcfg.lr, mcfg.weight_decay)
    scheduler = warmup_cosine(optimizer, int(tcfg.warmup_epochs * steps_per_epoch), epochs * steps_per_epoch)
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.label_smoothing)
    amp_dtype, scaler = amp_setup(device, tcfg.amp)
    stopper = EarlyStopping(tcfg.patience)
    logger = CSVLogger(run_dir / "metrics.csv")

    print(f"Model {args.model}: {mcfg.arch} ({n_params / 1e6:.1f}M params) on {device}, "
          f"amp={amp_dtype}, train/val/test = {', '.join(str(len(d)) for d in datasets.values())}")

    t_start = now()
    for epoch in range(1, epochs + 1):
        t0 = now()
        tr = train_one_epoch(model, loaders["train"], optimizer, scheduler, criterion, device,
                             amp_dtype, scaler, tcfg.grad_clip, args.limit_batches)
        va = evaluate(model, loaders["val"], device, amp_dtype, args.limit_batches)
        improved = stopper.step(va["top1"])
        if improved:
            save_checkpoint(run_dir / "best.pt", model, arch=mcfg.arch, class_names=class_names,
                            preprocess=pre, epoch=epoch, val_top1=va["top1"], model_key=args.model)
        dt = now() - t0
        logger.log({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                    **{f"train_{k}": round(v, 5) for k, v in tr.items()},
                    **{f"val_{k}": round(v, 5) for k, v in va.items()},
                    "epoch_seconds": round(dt, 1), "best": improved})
        print(f"epoch {epoch:>2}/{epochs}  train loss {tr['loss']:.3f} top1 {tr['top1']:.3f} | "
              f"val loss {va['loss']:.3f} top1 {va['top1']:.3f} top5 {va['top5']:.3f}  "
              f"[{fmt_time(dt)}]{'  *best*' if improved else ''}")
        if stopper.should_stop:
            print(f"Early stopping: no val top-1 improvement for {tcfg.patience} epochs.")
            break

    # Evaluate the *best* checkpoint (not the last epoch) on the held-out test split.
    best_model, ckpt = load_classifier(run_dir / "best.pt", device)
    logits, labels = predict(best_model, loaders["test"], device, amp_dtype, args.limit_batches, desc="test")
    metrics = save_test_outputs(run_dir, logits, labels, class_names, extra={
        "model": args.model, "arch": mcfg.arch, "params": n_params, "best_epoch": ckpt["epoch"],
        "best_val_top1": ckpt["val_top1"], "epochs_trained": len(stopper.history),
        "train_minutes": round((now() - t_start) / 60, 1), "device": str(device),
    })
    print(f"\nTest (best epoch {ckpt['epoch']}): top-1 {metrics['top1']:.4f}, top-5 {metrics['top5']:.4f}")
    print(f"Outputs in {run_dir}/")


if __name__ == "__main__":
    main()
