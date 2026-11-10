"""Latency benchmark: PyTorch vs ONNX Runtime (batch 1 by default, fp32, forward pass only).

Backends: PyTorch CPU, ONNX Runtime CPU (same thread count), plus PyTorch MPS/CUDA and
ONNX Runtime CoreML when available. Writes <reports_dir>/deploy/benchmark.{json,md}.

Usage:
    python scripts/benchmark.py [--models cnn vit] [--warmup 20] [--iters 100] [--batch-size 1]
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from food_vision.benchmark import measure_latency, torch_latency
from food_vision.config import DEFAULT_CONFIG, load_config
from food_vision.device import get_device
from food_vision.inference import OnnxClassifier
from food_vision.models import load_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+", default=["cnn", "vit"])
    parser.add_argument("--runs-dir")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    runs = Path(args.runs_dir or cfg.train.runs_dir)
    out = cfg.data.reports_dir / "deploy"
    out.mkdir(parents=True, exist_ok=True)
    threads = torch.get_num_threads()
    accel = get_device(cfg.train.device)
    bs, w, n = args.batch_size, args.warmup, args.iters

    rows = []
    for key in args.models:
        model, ckpt = load_classifier(runs / key / "best.pt", "cpu")
        size = ckpt["preprocess"].image_size
        x = np.random.default_rng(0).standard_normal((bs, 3, size, size), dtype=np.float32)

        def add(backend: str, stats: dict) -> None:
            rows.append({"model": key, "arch": ckpt["arch"], "backend": backend, **stats})
            print(f"  {key:<4} {backend:<26} mean {stats['mean_ms']:7.2f} ms   p95 {stats['p95_ms']:7.2f} ms")

        print(f"{key}: {ckpt['arch']}")
        add(f"PyTorch CPU ({threads} thr)", torch_latency(model, "cpu", size, bs, w, n))
        onnx_path = Path(f"models/{key}.onnx")
        if not onnx_path.exists():
            print(f"  (skipping ONNX: {onnx_path} missing; run scripts/export_onnx.py --model {key})")
        else:
            cpu = OnnxClassifier(onnx_path, ["CPUExecutionProvider"], threads=threads)
            add(f"ONNX Runtime CPU ({threads} thr)", measure_latency(lambda: cpu.logits(x), w, n))
            if "CoreMLExecutionProvider" in ort.get_available_providers():
                try:
                    cml = OnnxClassifier(onnx_path, ["CoreMLExecutionProvider", "CPUExecutionProvider"])
                    add("ONNX Runtime CoreML", measure_latency(lambda: cml.logits(x), w, n))
                except Exception as e:  # noqa: BLE001 - optional backend
                    print(f"  (CoreML EP failed: {type(e).__name__}: {e})")
        if accel.type != "cpu":
            add(f"PyTorch {accel.type.upper()}", torch_latency(model, accel, size, bs, w, n))
        del model

    df = pd.DataFrame(rows)
    env = {
        "platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
        "torch": torch.__version__, "onnxruntime": ort.__version__, "cpu_threads": threads,
        "torch_onednn_available": torch.backends.mkldnn.is_available(),
        "batch_size": bs, "warmup": w, "iters": n,
    }
    (out / "benchmark.json").write_text(json.dumps({"environment": env, "results": rows}, indent=2))

    table = df.pivot_table(index="backend", columns="model", values=["mean_ms", "p95_ms"], sort=False)
    table.columns = [f"{m} {stat.replace('_ms', '')} (ms)" for stat, m in table.columns]
    table = table[[c for m in args.models for c in table.columns if c.startswith(f"{m} ")]]
    md = [
        "# Inference latency: PyTorch vs ONNX Runtime", "",
        f"Batch {bs}, fp32, model forward pass only (no image decoding/preprocessing); "
        f"{w} warm-up runs, then mean / p95 over {n} runs.", "",
        table.round(2).to_markdown(), "",
        f"Environment: {env['platform']}, Python {env['python']}, torch {env['torch']}, "
        f"onnxruntime {env['onnxruntime']}, {threads} CPU threads, "
        f"torch oneDNN available: {env['torch_onednn_available']}.",
    ]
    (out / "benchmark.md").write_text("\n".join(md) + "\n")
    print("\n" + table.round(2).to_markdown())
    print(f"\nOutputs in {out}/")


if __name__ == "__main__":
    main()
