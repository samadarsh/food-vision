"""What does each model look at? Grad-CAM (CNNs) and attention rollout (ViTs)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from torch import nn


def default_cam_layer(model: nn.Module) -> nn.Module:
    """Last convolutional block of common torchvision CNNs."""
    if hasattr(model, "features"):  # EfficientNet / MobileNet / ConvNeXt
        return model.features[-1]
    if hasattr(model, "layer4"):  # ResNet
        return model.layer4
    raise ValueError(f"Pick a Grad-CAM layer for {type(model).__name__} manually")


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017).

    Channel weights = spatially averaged gradients of the class score w.r.t. the target
    layer's activations; the map is ReLU(sum_c weight_c * activation_c), upsampled.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model.eval()
        layer = target_layer or default_cam_layer(model)
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        self._acts = output
        if output.requires_grad:  # skip plain no-grad forwards
            output.register_hook(lambda g: setattr(self, "_grads", g))

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int, float]:
        """x: (1, 3, H, W). Returns (heatmap in [0, 1] of shape (H, W), explained class, its prob)."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            idx = int(logits.argmax(1)) if class_idx is None else class_idx
            logits[0, idx].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)          # (1, C, 1, 1)
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        prob = float(torch.softmax(logits.detach(), 1)[0, idx])
        return _normalize(cam[0, 0].detach().cpu().numpy()), idx, prob

    def remove(self) -> None:
        self._handle.remove()


class AttentionRollout:
    """Attention rollout (Abnar & Zuidema, 2020) for timm VisionTransformers.

    Per layer: average heads, add identity (residual connection), re-normalise rows;
    multiply across layers; the class token's row over patch tokens is the map.
    Class-agnostic: shows what the CLS token aggregates, not evidence for one class.
    """

    def __init__(self, model: nn.Module, head_fusion: str = "mean"):
        self.model = model.eval()
        self.head_fusion = head_fusion
        self.num_prefix = getattr(model, "num_prefix_tokens", 1)
        self._attn: list[torch.Tensor] = []
        self._handles = []
        for block in model.blocks:
            block.attn.fused_attn = False  # fused SDPA never materialises the attention matrix
            self._handles.append(block.attn.attn_drop.register_forward_hook(
                lambda m, i, o: self._attn.append(o.detach())))

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, int, float]:
        """x: (1, 3, H, W). Returns (heatmap in [0, 1] of shape (H, W), predicted class, its prob)."""
        self._attn.clear()
        probs = torch.softmax(self.model(x), 1)[0]
        pred = int(probs.argmax())
        rollout = None
        for a in self._attn:  # (1, heads, N, N)
            a = a.max(dim=1).values if self.head_fusion == "max" else a.mean(dim=1)
            a = a + torch.eye(a.shape[-1], device=a.device)
            a = a / a.sum(dim=-1, keepdim=True)
            rollout = a if rollout is None else a @ rollout
        cls_to_patches = rollout[0, 0, self.num_prefix:]
        side = int(cls_to_patches.numel() ** 0.5)
        grid = cls_to_patches.reshape(1, 1, side, side)
        heat = F.interpolate(grid, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return _normalize(heat[0, 0].cpu().numpy()), pred, float(probs[pred])

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


def _normalize(a: np.ndarray) -> np.ndarray:
    a = a - a.min()
    return a / a.max() if a.max() > 0 else a


def denormalize(x: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> np.ndarray:
    """(3, H, W) normalised tensor -> (H, W, 3) float image in [0, 1]."""
    img = x.detach().cpu() * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45, cmap: str = "jet") -> np.ndarray:
    """Blend a [0, 1] heatmap (H, W) onto an RGB image (H, W, 3) in [0, 1]."""
    colored = colormaps[cmap](heatmap)[..., :3]
    return np.clip((1 - alpha) * image + alpha * colored, 0.0, 1.0)


@torch.no_grad()
def artifact_token_check(rollout: AttentionRollout, images: list[torch.Tensor], layer: int,
                         k: int = 5) -> dict[str, float]:
    """Do rollout hot spots coincide with high-norm 'artifact' tokens (Darcet et al., 2023)?

    Measures, over `images`, the patch-token norms in the residual stream after block
    `layer` (before the final LayerNorm, which would hide them): the max/median norm
    ratio, and how many of rollout's top-k patches are also top-k by norm.
    """
    model, captured = rollout.model, {}
    handle = model.blocks[layer].register_forward_hook(
        lambda m, i, o: captured.__setitem__("x", o[0, rollout.num_prefix:]))
    ratios, overlaps, corrs = [], [], []
    try:
        for x in images:
            heat_tokens = _rollout_tokens(rollout, x.unsqueeze(0))
            norms = captured["x"].norm(dim=-1)
            ratios.append(float(norms.max() / norms.median()))
            top_r, top_n = set(heat_tokens.topk(k).indices.tolist()), set(norms.topk(k).indices.tolist())
            overlaps.append(len(top_r & top_n) / k)
            corrs.append(float(np.corrcoef(heat_tokens.cpu().numpy(), norms.cpu().numpy())[0, 1]))
    finally:
        handle.remove()
    n_patches = int(norms.numel())
    return {"layer": layer, "num_images": len(images), "top_k": k,
            "median_max_over_median_norm": float(np.median(ratios)),
            "mean_topk_overlap": float(np.mean(overlaps)),
            "chance_topk_overlap": k / n_patches,
            "mean_corr_rollout_vs_norm": float(np.mean(corrs))}


def _rollout_tokens(rollout: AttentionRollout, x: torch.Tensor) -> torch.Tensor:
    """Rollout CLS->patch weights (flat, one per patch token) for one image."""
    rollout._attn.clear()
    rollout.model(x)
    r = None
    for a in rollout._attn:
        a = a.mean(dim=1) + torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        r = a if r is None else a @ r
    return r[0, 0, rollout.num_prefix:]
