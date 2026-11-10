"""Device selection: Apple MPS -> CUDA -> CPU."""

from __future__ import annotations

import torch


def get_device(preferred: str = "auto") -> torch.device:
    """Return the best available device, or `preferred` if it is set and available."""
    if preferred != "auto":
        device = torch.device(preferred)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
