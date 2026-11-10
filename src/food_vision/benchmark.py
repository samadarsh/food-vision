"""Inference latency measurement (reused for PyTorch vs ONNX Runtime in Phase 4)."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch


def synchronize(device: torch.device | str) -> None:
    """Wait for queued GPU work so timings measure compute, not just kernel launch."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_latency(fn: Callable[[], object], warmup: int = 10, iters: int = 50,
                    sync: Callable[[], None] | None = None) -> dict[str, float]:
    """Time `fn()` `iters` times after `warmup` calls. Returns mean/p50/p95/min in milliseconds."""
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)),
            "p95_ms": float(np.percentile(t, 95)), "min_ms": float(t.min()), "iters": iters}


@torch.no_grad()
def torch_latency(model: torch.nn.Module, device: torch.device | str, image_size: int = 224,
                  batch_size: int = 1, warmup: int = 10, iters: int = 50) -> dict[str, float]:
    """Latency of a model's forward pass on `device` (fp32, eval mode, random input)."""
    device = torch.device(device)
    model = model.to(device).eval()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    return measure_latency(lambda: model(x), warmup, iters, sync=lambda: synchronize(device))
