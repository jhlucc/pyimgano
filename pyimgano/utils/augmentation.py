"""Augmentation registry and pipeline utilities."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from PIL import Image
from PIL import ImageEnhance

from .image_ops import random_horizontal_flip
from .image_ops_cv import (
    add_gaussian_noise,
    adjust_brightness_contrast,
    gaussian_blur,
    motion_blur,
    random_brightness_contrast,
    random_crop_and_resize,
    random_gaussian_noise,
    random_rotation,
)

try:  # optional torchvision support
    import torchvision.transforms as T
    _TORCHVISION_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    T = None
    _TORCHVISION_AVAILABLE = False

try:  # diffusion support
    from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
    _DIFFUSERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    StableDiffusionPipeline = None
    StableDiffusionImg2ImgPipeline = None
    _DIFFUSERS_AVAILABLE = False


AugmentationFn = Callable[[Any], Any]


class AugmentationRegistry:
    """Registry for augmentation factories/functions."""

    def __init__(self) -> None:
        self._store: MutableMapping[str, tuple[Callable[..., AugmentationFn], bool]] = {}

    def register(self, name: str, factory: Callable[..., AugmentationFn], *, override: bool = False,
                 is_factory: bool = True) -> None:
        if not override and name in self._store:
            raise KeyError(f"Augmentation {name!r} already registered")
        self._store[name] = (factory, is_factory)

    def build(self, name: str, **kwargs) -> AugmentationFn:
        factory, is_factory = self._store[name]
        if is_factory:
            return factory(**kwargs)
        if kwargs:
            return partial(factory, **kwargs)  # type: ignore[arg-type]
        return factory  # type: ignore[return-value]

    def get(self, name: str) -> AugmentationFn:
        return self.build(name)

    def list_available(self) -> Sequence[str]:
        return sorted(self._store.keys())


AUGMENTATION_REGISTRY = AugmentationRegistry()


def register_augmentation(name: str, factory: Callable[..., AugmentationFn], *, override: bool = False,
                          is_factory: bool = True) -> None:
    AUGMENTATION_REGISTRY.register(name, factory, override=override, is_factory=is_factory)


def list_augmentations() -> Sequence[str]:
    return AUGMENTATION_REGISTRY.list_available()


def _wrap_pil_function(func: Callable[..., Image.Image]) -> Callable[..., AugmentationFn]:
    def factory(**kwargs) -> AugmentationFn:
        def augmentation(data):
            is_pil = isinstance(data, Image.Image)
            img = data if is_pil else Image.fromarray(data)
            result = func(img, **kwargs)
            return result if is_pil else np.asarray(result)
        return augmentation
    return factory


def _wrap_numpy_function(func: Callable[..., np.ndarray]) -> Callable[..., AugmentationFn]:
    def factory(**kwargs) -> AugmentationFn:
        def augmentation(data):
            is_pil = isinstance(data, Image.Image)
            array = np.asarray(data) if is_pil else data
            result = func(array, **kwargs)
            return result if not is_pil else Image.fromarray(result)
        return augmentation
    return factory


def _wrap_torchvision_transform(transform_cls: Callable, *, to_pil: bool = True) -> Callable[..., AugmentationFn]:
    if not _TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required for this augmentation")

    def factory(**kwargs) -> AugmentationFn:
        transform = transform_cls(**kwargs)

        def augmentation(data):
            is_pil = isinstance(data, Image.Image)
            if not is_pil:
                img = Image.fromarray(data)
            else:
                img = data
            result = transform(img)
            if isinstance(result, Image.Image):
                return result if is_pil else np.asarray(result)
            # torchvision transforms may return tensors
            np_result = np.asarray(np.clip(np.array(result), 0, 255), dtype=np.uint8)
            return Image.fromarray(np_result) if is_pil else np_result

        return augmentation

    return factory


def random_cutout(image: np.ndarray, max_frac: float = 0.4, fill: int = 0) -> np.ndarray:
    """Apply random cutout mask to image."""

    if image.ndim == 2:
        h, w = image.shape
        c = None
    else:
        h, w, c = image.shape
    cut_h = int(h * np.random.uniform(0.05, max_frac))
    cut_w = int(w * np.random.uniform(0.05, max_frac))
    top = np.random.randint(0, max(1, h - cut_h + 1))
    left = np.random.randint(0, max(1, w - cut_w + 1))
    out = image.copy()
    if c is None:
        out[top:top + cut_h, left:left + cut_w] = fill
    else:
        out[top:top + cut_h, left:left + cut_w, :] = fill
    return out


def mixup_batch(images: np.ndarray, labels: np.ndarray, alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Perform mixup on batch of images and labels."""

    if alpha <= 0:
        return images, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    indices = np.random.permutation(images.shape[0])
    mixed_images = lam * images + (1 - lam) * images[indices]
    mixed_labels = lam * labels + (1 - lam) * labels[indices]
    return mixed_images, mixed_labels, lam


def cutmix_batch(images: np.ndarray, labels: np.ndarray, alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Perform CutMix augmentation on batch."""

    if alpha <= 0:
        return images, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    indices = np.random.permutation(images.shape[0])
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]

    if images.ndim != 4:
        raise ValueError("cutmix_batch expects images with shape (N, C, H, W) or (N, H, W, C)")
    if images.shape[1] <= 4:  # assume NCHW
        channels_first = True
        h, w = images.shape[2], images.shape[3]
    else:  # assume NHWC
        channels_first = False
        h, w = images.shape[1], images.shape[2]

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)

    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)

    mixed_images = images.copy()
    if channels_first:
        mixed_images[:, :, y1:y2, x1:x2] = shuffled_images[:, :, y1:y2, x1:x2]
    else:
        mixed_images[:, y1:y2, x1:x2, :] = shuffled_images[:, y1:y2, x1:x2, :]

    lam_adjusted = 1 - ((y2 - y1) * (x2 - x1) / (h * w))
    mixed_labels = lam_adjusted * labels + (1 - lam_adjusted) * shuffled_labels
    return mixed_images, mixed_labels, lam_adjusted


def _enhance(image: Image.Image, op: str, magnitude: float) -> Image.Image:
    enhancer_map = {
        "brightness": ImageEnhance.Brightness,
        "contrast": ImageEnhance.Contrast,
        "color": ImageEnhance.Color,
        "sharpness": ImageEnhance.Sharpness,
    }
    enhancer_cls = enhancer_map[op]
    enhancer = enhancer_cls(image)
    return enhancer.enhance(magnitude)


def trivial_augment(image: Image.Image, magnitude: float = 0.5) -> Image.Image:
    """Single-operation TrivialAugment."""

    ops = [
        lambda img, mag: img.rotate(int(np.random.uniform(-30, 30)), resample=Image.BILINEAR),
        lambda img, mag: img.transform(img.size, Image.AFFINE, (1, mag * np.random.uniform(-0.3, 0.3), 0, 0, 1, 0)),
        lambda img, mag: _enhance(img, "brightness", 1 + (np.random.uniform(-1, 1) * mag)),
        lambda img, mag: _enhance(img, "contrast", 1 + (np.random.uniform(-1, 1) * mag)),
        lambda img, mag: _enhance(img, "color", 1 + (np.random.uniform(-1, 1) * mag)),
        lambda img, mag: _enhance(img, "sharpness", 1 + (np.random.uniform(-1, 1) * mag)),
    ]
    operation = random_horizontal_flip if np.random.rand() < 0.1 else random.choice(ops)
    if operation is random_horizontal_flip:
        return operation(image)
    return operation(image, magnitude)


def color_jitter(image: Image.Image | np.ndarray, brightness: float = 0.4, contrast: float = 0.4,
                 saturation: float = 0.4, hue: float = 0.1) -> Image.Image | np.ndarray:
    if _TORCHVISION_AVAILABLE:
        transform = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        is_pil = isinstance(image, Image.Image)
        img = image if is_pil else Image.fromarray(image)
        result = transform(img)
        return result if is_pil else np.asarray(result)

    is_pil = isinstance(image, Image.Image)
    img = image if is_pil else Image.fromarray(image)
    def jitter(enhance_cls, magnitude):
        if magnitude <= 0:
            return img
        factor = 1 + np.random.uniform(-magnitude, magnitude)
        enhancer = enhance_cls(img)
        return enhancer.enhance(max(0, factor))

    img = jitter(ImageEnhance.Brightness, brightness)
    img = jitter(ImageEnhance.Contrast, contrast)
    img = jitter(ImageEnhance.Color, saturation)
    if not is_pil:
        return np.asarray(img)
    return img


def _autoaugment_policy(name: str):
    if not _TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required for AutoAugment")
    name = name.lower()
    mapping = {
        "imagenet": T.AutoAugmentPolicy.IMAGENET,
        "cifar10": T.AutoAugmentPolicy.CIFAR10,
        "svhn": T.AutoAugmentPolicy.SVHN,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported AutoAugment policy {name!r}")
    return mapping[name]


def auto_augment(image: Image.Image | np.ndarray, policy: str = "imagenet") -> Image.Image | np.ndarray:
    if not _TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required for AutoAugment")
    transform = T.AutoAugment(policy=_autoaugment_policy(policy))
    is_pil = isinstance(image, Image.Image)
    img = image if is_pil else Image.fromarray(image)
    result = transform(img)
    return result if is_pil else np.asarray(result)


def rand_augment(image: Image.Image | np.ndarray, num_ops: int = 2, magnitude: int = 9) -> Image.Image | np.ndarray:
    if not _TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required for RandAugment")
    transform = T.RandAugment(num_ops=num_ops, magnitude=magnitude)
    is_pil = isinstance(image, Image.Image)
    img = image if is_pil else Image.fromarray(image)
    result = transform(img)
    return result if is_pil else np.asarray(result)

def resolve_augmentation(spec: Any, registry: AugmentationRegistry | None = None) -> AugmentationFn:
    registry = registry or AUGMENTATION_REGISTRY
    if callable(spec) and not isinstance(spec, str):
        return spec  # already augmentation
    if isinstance(spec, str):
        return registry.get(spec)
    if isinstance(spec, Mapping):
        name = spec.get("name")
        if name is None:
            raise ValueError("Augmentation spec dict must contain 'name'")
        kwargs = spec.get("kwargs", {})
        return registry.build(name, **kwargs)
    raise TypeError(f"Unsupported augmentation spec: {spec!r}")


def build_augmentation_pipeline(augmentations: Sequence[Any], *, return_type: str = "same",
                                registry: AugmentationRegistry | None = None) -> AugmentationFn:
    registry = registry or AUGMENTATION_REGISTRY
    resolved = [resolve_augmentation(spec, registry=registry) for spec in augmentations]

    def pipeline(data):
        initial_pil = isinstance(data, Image.Image)
        current = data
        for aug in resolved:
            current = aug(current)

        if return_type == "pil":
            if isinstance(current, np.ndarray):
                current = Image.fromarray(current)
        elif return_type == "numpy":
            if isinstance(current, Image.Image):
                current = np.asarray(current)
        elif return_type == "same":
            if initial_pil and isinstance(current, np.ndarray):
                current = Image.fromarray(current)
            elif not initial_pil and isinstance(current, Image.Image):
                current = np.asarray(current)
        else:
            raise ValueError("return_type must be one of {'same','pil','numpy'}")
        return current

    return pipeline


def _register_default_augmentations() -> None:
    register_augmentation("random_horizontal_flip", _wrap_pil_function(random_horizontal_flip))
    register_augmentation("random_rotation", _wrap_numpy_function(random_rotation))
    register_augmentation("random_crop_resize", _wrap_numpy_function(random_crop_and_resize))
    register_augmentation("gaussian_noise", _wrap_numpy_function(add_gaussian_noise))
    register_augmentation("random_gaussian_noise", _wrap_numpy_function(random_gaussian_noise))
    register_augmentation("brightness_contrast", _wrap_numpy_function(adjust_brightness_contrast))
    register_augmentation("random_brightness_contrast", _wrap_numpy_function(random_brightness_contrast))
    register_augmentation("gaussian_blur", _wrap_numpy_function(gaussian_blur))
    register_augmentation("motion_blur", _wrap_numpy_function(motion_blur))
    register_augmentation("random_cutout", _wrap_numpy_function(random_cutout))
    register_augmentation("trivial_augment", _wrap_pil_function(trivial_augment))
    register_augmentation("color_jitter", _wrap_pil_function(color_jitter))
    if _TORCHVISION_AVAILABLE:
        register_augmentation("auto_augment", _wrap_pil_function(auto_augment), override=True)
        register_augmentation("rand_augment", _wrap_pil_function(rand_augment), override=True)


_register_default_augmentations()


class DiffusionAugmentor:
    """Wrapper around Stable Diffusion pipelines for data augmentation."""

    def __init__(
        self,
        model: str = "runwayml/stable-diffusion-v1-5",
        *,
        device: str | None = None,
        pipeline: str = "txt2img",
        dtype: str | None = "float16",
        **kwargs,
    ) -> None:
        if not _DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers is required for DiffusionAugmentor")

        self.pipeline_type = pipeline
        self.device = device

        if pipeline == "txt2img":
            self.pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=_resolve_torch_dtype(dtype), **kwargs)
        elif pipeline == "img2img":
            if StableDiffusionImg2ImgPipeline is None:
                raise ImportError("StableDiffusionImg2ImgPipeline not available; update diffusers")
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model, torch_dtype=_resolve_torch_dtype(dtype), **kwargs)
        else:
            raise ValueError("pipeline must be 'txt2img' or 'img2img'")

        if device is not None:
            self.pipe.to(device)
        self.pipe.safety_checker = None

    def generate(self, prompt: str, *, num_images: int = 1, guidance_scale: float = 7.5,
                 height: int | None = None, width: int | None = None, **kwargs) -> list[Image.Image]:
        if self.pipeline_type != "txt2img":
            raise RuntimeError("generate is only available for txt2img pipeline")
        result = self.pipe(prompt=prompt, num_images_per_prompt=num_images, guidance_scale=guidance_scale,
                           height=height, width=width, **kwargs)
        return result.images

    def augment(self, image: Image.Image, prompt: str, *, strength: float = 0.75, guidance_scale: float = 7.5,
                num_images: int = 1, **kwargs) -> list[Image.Image]:
        if self.pipeline_type != "img2img":
            raise RuntimeError("augment requires img2img pipeline")
        result = self.pipe(prompt=prompt, image=image, strength=strength, guidance_scale=guidance_scale,
                           num_images_per_prompt=num_images, **kwargs)
        return result.images

    def __call__(self, *args, **kwargs):
        if self.pipeline_type == "txt2img":
            return self.generate(*args, **kwargs)
        return self.augment(*args, **kwargs)


def _resolve_torch_dtype(dtype: str | None):
    if dtype is None:
        return None
    import torch

    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if isinstance(dtype, str):
        return mapping.get(dtype.lower(), torch.float16)
    return dtype


__all__ = [
    "AUGMENTATION_REGISTRY",
    "register_augmentation",
    "list_augmentations",
    "resolve_augmentation",
    "build_augmentation_pipeline",
    "mixup_batch",
    "cutmix_batch",
    "random_cutout",
    "trivial_augment",
    "color_jitter",
    "auto_augment",
    "rand_augment",
    "DiffusionAugmentor",
]
