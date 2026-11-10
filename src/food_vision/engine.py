"""Training / evaluation loops: AMP, warmup+cosine LR, early stopping, CSV logging."""

from __future__ import annotations

import contextlib
import csv
import math
import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from food_vision.metrics import topk_accuracy, topk_correct

# With fp16 GradScaler the first optimizer step(s) are skipped by design when the initial
# loss scale overflows; PyTorch then warns that the LR scheduler stepped "before" the
# optimizer. The effect is a one-step offset in the schedule, so the warning is noise.
warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler.step\(\)` before")

# --------------------------------------------------------------------------- mixed precision


def amp_setup(device: torch.device, enabled: bool) -> tuple[torch.dtype | None, torch.amp.GradScaler | None]:
    """Choose autocast dtype + grad scaler for the device. (None, None) means full fp32.

    CUDA: bf16 if supported (no scaler needed), else fp16 + GradScaler.
    MPS:  fp16 + GradScaler. CPU: disabled (CPU autocast rarely speeds up training).
    """
    if not enabled or device.type == "cpu":
        return None, None
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler(device.type)


def autocast(device: torch.device, dtype: torch.dtype | None) -> contextlib.AbstractContextManager:
    """Autocast context, or a no-op when `dtype` is None."""
    return torch.autocast(device.type, dtype=dtype) if dtype else contextlib.nullcontext()


# --------------------------------------------------------------------------- optimiser / schedule


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW without weight decay on biases and norm parameters (standard practice)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def warmup_cosine(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int,
                  min_ratio: float = 0.01) -> LambdaLR:
    """Per-step schedule: linear warmup to the base LR, then cosine decay to `min_ratio` x base."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)


# --------------------------------------------------------------------------- loops


def _limited(loader: DataLoader, limit: int | None) -> Iterator:
    for i, batch in enumerate(loader):
        if limit is not None and i >= limit:
            return
        yield batch


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scheduler: LambdaLR, criterion: nn.Module, device: torch.device,
                    amp_dtype: torch.dtype | None, scaler: torch.amp.GradScaler | None,
                    grad_clip: float | None, limit_batches: int | None = None) -> dict[str, float]:
    """One pass over the training data. Returns mean loss and top-1/top-5 accuracy."""
    model.train()
    loss_sum, n, correct = 0.0, 0, {1: 0, 5: 0}
    total = min(len(loader), limit_batches or len(loader))
    for x, y in tqdm(_limited(loader, limit_batches), total=total, desc="train", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, amp_dtype):
            logits = model(x)
            loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss {loss.item()}; try train.amp: false or a lower lr")

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        loss_sum += loss.item() * len(y)
        n += len(y)
        for k, c in topk_correct(logits.detach().float(), y).items():
            correct[k] += c
    return {"loss": loss_sum / n, "top1": correct[1] / n, "top5": correct[5] / n}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device,
            amp_dtype: torch.dtype | None = None, limit_batches: int | None = None,
            desc: str = "eval") -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model over a loader. Returns (logits, labels) on CPU, in loader order."""
    model.eval()
    all_logits, all_labels = [], []
    total = min(len(loader), limit_batches or len(loader))
    for x, y in tqdm(_limited(loader, limit_batches), total=total, desc=desc, leave=False):
        with autocast(device, amp_dtype):
            logits = model(x.to(device, non_blocking=True))
        all_logits.append(logits.float().cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             amp_dtype: torch.dtype | None = None, limit_batches: int | None = None) -> dict[str, float]:
    """Plain cross-entropy loss (no smoothing) + top-1/top-5 on a loader."""
    logits, labels = predict(model, loader, device, amp_dtype, limit_batches)
    return {"loss": nn.functional.cross_entropy(logits, labels).item(), **topk_accuracy(logits, labels)}


# --------------------------------------------------------------------------- bookkeeping


@dataclass
class EarlyStopping:
    """Stop when the monitored value (higher = better) has not improved for `patience` epochs."""

    patience: int
    best: float = -math.inf
    bad_epochs: int = 0
    history: list[float] = field(default_factory=list)

    def step(self, value: float) -> bool:
        """Record a value; return True if it is a new best."""
        self.history.append(value)
        if value > self.best:
            self.best, self.bad_epochs = value, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


class CSVLogger:
    """Append one dict per row to a CSV file (header written on first row)."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        self._fields: list[str] | None = None

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            if self._fields is None:
                self._fields = list(row)
                csv.DictWriter(f, self._fields).writeheader()
            csv.DictWriter(f, self._fields).writerow(row)


def fmt_time(seconds: float) -> str:
    """Seconds as m:ss."""
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def now() -> float:
    """Wall-clock timer (kept here so the scripts don't import time just for this)."""
    return time.perf_counter()
