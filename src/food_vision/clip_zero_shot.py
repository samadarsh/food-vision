"""Zero-shot classification with OpenCLIP: no training, class names become text prompts."""

from __future__ import annotations

import torch
from torch import nn


def class_to_text(name: str) -> str:
    """Food-101 folder name -> natural text, e.g. 'chicken_curry' -> 'chicken curry'."""
    return name.replace("_", " ")


class ZeroShotCLIP(nn.Module):
    """CLIP image encoder + a fixed text-embedding classifier head.

    Each class embedding is the mean of its L2-normalised prompt embeddings over all
    templates ("prompt ensembling", as in the CLIP paper), re-normalised. Logits are
    scaled cosine similarities, so it plugs into the same evaluation code as the
    trained classifiers.
    """

    def __init__(self, arch: str, pretrained: str, class_names: list[str], templates: list[str],
                 device: torch.device):
        super().__init__()
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model = model.to(device).eval()
        self.preprocess = preprocess  # CLIP's own eval transform (resize, crop, CLIP mean/std)
        self.logit_scale = float(model.logit_scale.detach().exp())
        tokenizer = open_clip.get_tokenizer(arch)

        with torch.no_grad():
            weights = []
            for name in class_names:
                prompts = [t.format(class_to_text(name)) for t in templates]
                emb = self.model.encode_text(tokenizer(prompts).to(device))
                emb = emb / emb.norm(dim=-1, keepdim=True)
                mean = emb.mean(dim=0)
                weights.append(mean / mean.norm())
        self.register_buffer("text_weights", torch.stack(weights, dim=1))  # (dim, num_classes)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return self.logit_scale * feats @ self.text_weights
