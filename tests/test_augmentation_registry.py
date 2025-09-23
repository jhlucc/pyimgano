import importlib

import numpy as np
import pytest
from PIL import Image

from pyimgano.utils import (
    AUGMENTATION_REGISTRY,
    build_augmentation_pipeline,
    list_augmentations,
)


def test_trivial_augment_registered():
    aug_names = list_augmentations()
    assert "trivial_augment" in aug_names


def test_trivial_augment_executes_without_error():
    pipeline = build_augmentation_pipeline(["trivial_augment"], return_type="pil")
    img = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
    result = pipeline(img)
    assert isinstance(result, Image.Image)


def test_color_jitter_pipeline_numpy_output():
    pipeline = build_augmentation_pipeline(["color_jitter"], return_type="numpy")
    img = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
    result = pipeline(img)
    assert isinstance(result, np.ndarray)
    assert result.shape == (32, 32, 3)


def test_mixup_cutmix_helpers():
    from pyimgano.utils.augmentation import mixup_batch, cutmix_batch

    images = np.random.rand(4, 3, 16, 16).astype(np.float32)
    labels = np.eye(4).astype(np.float32)

    mixed_images, mixed_labels, lam = mixup_batch(images, labels, alpha=0.4)
    assert mixed_images.shape == images.shape
    assert mixed_labels.shape == labels.shape
    assert 0 <= lam <= 1

    mixed_images, mixed_labels, lam = cutmix_batch(images, labels, alpha=0.4)
    assert mixed_images.shape == images.shape
    assert mixed_labels.shape == labels.shape
    assert 0 <= lam <= 1


TORCHVISION_AVAILABLE = importlib.util.find_spec("torchvision") is not None


@pytest.mark.skipif(not TORCHVISION_AVAILABLE, reason="requires torchvision")
def test_auto_augment_pipeline():
    pipeline = build_augmentation_pipeline(["auto_augment"], return_type="pil")
    img = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
    result = pipeline(img)
    assert isinstance(result, Image.Image)


@pytest.mark.skipif(not TORCHVISION_AVAILABLE, reason="requires torchvision")
def test_rand_augment_pipeline():
    pipeline = build_augmentation_pipeline(["rand_augment"], return_type="pil")
    img = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
    result = pipeline(img)
    assert isinstance(result, Image.Image)


def test_diffusion_augmentor_optional():
    from pyimgano.utils.augmentation import DiffusionAugmentor, _DIFFUSERS_AVAILABLE

    if _DIFFUSERS_AVAILABLE:
        pytest.skip("diffusers available; skipping heavy pipeline test")
    with pytest.raises(ImportError):
        DiffusionAugmentor()
