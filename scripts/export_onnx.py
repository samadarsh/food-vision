"""Export a trained classifier to ONNX and verify it against PyTorch on the full test split.

Three independent checks (written to <reports_dir>/deploy/onnx_verify_<model>.json):
  1. Conversion: identical input tensors -> PyTorch vs ONNX Runtime logits.
  2. Preprocessing: torch-free NumPy/PIL preprocessing (used by the API) vs torchvision.
  3. End to end: ONNX Runtime + NumPy preprocessing top-1 vs the Phase 2 PyTorch result.

Usage:
    python scripts/export_onnx.py --model vit [--out models/vit.onnx] [--atol 1e-3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from food_vision.config import DEFAULT_CONFIG, load_config
from food_vision.data import FoodDataset, build_transforms, food101_dir, get_splits
from food_vision.export import compare_logits, export_onnx
from food_vision.inference import OnnxClassifier, preprocess
from food_vision.models import load_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="vit", help="run key: cnn or vit")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--runs-dir")
    parser.add_argument("--out", help="default: models/<model>.onnx")
    parser.add_argument("--atol", type=float, default=1e-3, help="max allowed |logit| difference")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    runs = Path(args.runs_dir or cfg.train.runs_dir)
    out = Path(args.out or f"models/{args.model}.onnx")
    report_dir = cfg.data.reports_dir / "deploy"
    report_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_classifier(runs / args.model / "best.pt", "cpu")
    pre = ckpt["preprocess"]
    resize = int(round(pre.image_size / 0.875))  # must match data.build_transforms
    print(f"Exporting {ckpt['arch']} -> {out}")
    export_onnx(model, out, image_size=pre.image_size, resize_size=resize, mean=pre.mean, std=pre.std,
                class_names=ckpt["class_names"], arch=ckpt["arch"])
    print(f"  {out.stat().st_size / 1e6:.1f} MB")

    onnx_model = OnnxClassifier(out)
    test = get_splits(cfg)["test"]
    images_dir = food101_dir(cfg.data.root) / "images"
    ds = FoodDataset(test, images_dir, build_transforms(pre.image_size, False, pre.mean, pre.std))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=cfg.data.num_workers)

    # 1. Conversion check on identical tensors.
    t_logits, o_logits, labels = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="PyTorch vs ONNX", leave=False):
            t_logits.append(model(x).numpy())
            o_logits.append(onnx_model.logits(x.numpy()))
            labels.append(y.numpy())
    labels_np = np.concatenate(labels)
    conversion = compare_logits(np.concatenate(t_logits), np.concatenate(o_logits), labels_np)

    # 2 + 3. NumPy preprocessing vs torchvision, and the full deployed path.
    pre_diffs, e2e_logits = [], []
    for i in tqdm(range(len(ds)), desc="deployed path", leave=False):
        x_torch, _ = ds[i]
        with Image.open(ds.path(i)) as im:
            x_np = preprocess(im, onnx_model.spec)
        pre_diffs.append(float(np.abs(x_np[0] - x_torch.numpy()).max()))
        e2e_logits.append(onnx_model.logits(x_np)[0])
    e2e_top1 = float((np.stack(e2e_logits).argmax(1) == labels_np).mean())
    recorded = json.loads((runs / args.model / "test_metrics.json").read_text())["top1"]

    report = {
        "model": args.model, "arch": ckpt["arch"], "onnx_path": str(out),
        "onnx_size_mb": round(out.stat().st_size / 1e6, 1), "atol": args.atol,
        "conversion": conversion,
        "preprocessing_max_abs_diff": max(pre_diffs),
        "preprocessing_mean_of_max_abs_diff": float(np.mean(pre_diffs)),
        "end_to_end_onnx_top1": e2e_top1,
        "recorded_pytorch_top1": recorded,
        "note": "recorded_pytorch_top1 comes from Phase 2 (fp16 autocast on the training device); "
                "conversion.torch_top1 is PyTorch fp32 on CPU, the like-for-like comparison.",
    }
    report["passed"] = bool(conversion["max_abs_logit_diff"] <= args.atol
                            and conversion["top1_agreement"] == 1.0)
    (report_dir / f"onnx_verify_{args.model}.json").write_text(json.dumps(report, indent=2))

    c = conversion
    print(f"Conversion ({c['num_images']} test images): max |logit diff| {c['max_abs_logit_diff']:.2e}, "
          f"top-1 agreement {c['top1_agreement']:.2%}, top-1 torch {c['torch_top1']:.4f} / onnx {c['onnx_top1']:.4f}")
    print(f"Preprocessing NumPy vs torchvision: max abs diff {report['preprocessing_max_abs_diff']:.2e} "
          f"(mean of per-image max {report['preprocessing_mean_of_max_abs_diff']:.2e})")
    print(f"End to end (ONNX + NumPy preprocessing): top-1 {e2e_top1:.4f} "
          f"(PyTorch fp32 {c['torch_top1']:.4f}; Phase 2 fp16-autocast record {recorded:.4f})")
    print("PASSED" if report["passed"] else "FAILED", f"-> {report_dir / f'onnx_verify_{args.model}.json'}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
