"""Training/eval transforms and the torch-free preprocessing used in deployment."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image

from food_vision.data import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from food_vision.inference import PreprocessSpec, preprocess, softmax

from .conftest import jpeg_bytes


def _img(size=(300, 200), mode="RGB") -> Image.Image:
    return Image.open(io.BytesIO(jpeg_bytes(size, mode)))


@pytest.mark.parametrize("train", [True, False])
def test_output_shape_and_dtype(train: bool) -> None:
    x = build_transforms(224, train=train)(_img((500, 375)))
    assert x.shape == (3, 224, 224)
    assert x.dtype == torch.float32


def test_eval_transform_is_deterministic() -> None:
    tf, img = build_transforms(224, train=False), _img()
    assert torch.equal(tf(img), tf(img))


def test_train_transform_is_random() -> None:
    tf, img = build_transforms(224, train=True), _img((512, 384))
    torch.manual_seed(0)
    a = tf(img)
    assert any(not torch.equal(a, tf(img)) for _ in range(5))


def test_normalisation_uses_given_mean_std() -> None:
    white = Image.new("RGB", (256, 256), (255, 255, 255))
    x = build_transforms(224, False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(white)
    assert torch.allclose(x, torch.ones_like(x))


@pytest.mark.parametrize("size,mode", [
    ((512, 384), "RGB"),   # landscape
    ((384, 512), "RGB"),   # portrait
    ((512, 512), "RGB"),   # square
    ((300, 187), "RGB"),   # odd sizes, shorter side < resize size (upsampling)
    ((400, 300), "L"),     # greyscale
    ((350, 280), "RGBA"),  # alpha channel
    ((309, 300), "RGB"),   # crop offset 19.5 -> round() gives 20, floor gives 19
    ((300, 309), "RGB"),   # same, vertical
])
def test_numpy_preprocess_matches_torchvision(size, mode) -> None:
    """The API's preprocessing must equal the training-time eval transform exactly."""
    img = _img(size, mode)
    spec = PreprocessSpec(224, 256, IMAGENET_MEAN, IMAGENET_STD)
    expected = build_transforms(224, False, IMAGENET_MEAN, IMAGENET_STD)(img.convert("RGB")).numpy()
    got = preprocess(img, spec)
    assert got.shape == (1, 3, 224, 224) and got.dtype == np.float32
    np.testing.assert_allclose(got[0], expected, atol=1e-6)


def test_softmax_rows_sum_to_one_and_are_stable() -> None:
    p = softmax(np.array([[1000.0, 1000.0, -1000.0], [0.0, 1.0, 2.0]]))
    assert np.allclose(p.sum(axis=1), 1.0) and np.isfinite(p).all()
    assert np.isclose(p[0, 0], 0.5)
