"""Common image preprocessing utilities for PyImgAno."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    Resampling = Image.Resampling  # Pillow >= 9.1
except AttributeError:  # pragma: no cover - for older Pillow
    class Resampling:
        NEAREST = Image.NEAREST
        BILINEAR = Image.BILINEAR
        BICUBIC = Image.BICUBIC
        LANCZOS = Image.LANCZOS


def load_image(path: str, mode: str = "RGB") -> Image.Image:
    """Load an image from disk and optionally convert color mode."""

    image = Image.open(path)
    if mode is not None:
        image = image.convert(mode)
    return image


def resize_image(image: Image.Image, size: Tuple[int, int], *, keep_ratio: bool = False,
                 resample=Resampling.BILINEAR) -> Image.Image:
    """Resize image to a target size; optionally preserve aspect ratio."""

    if keep_ratio:
        image.thumbnail(size, resample=resample)
        return image
    return image.resize(size, resample=resample)


def center_crop(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Return central crop of specified size."""

    width, height = image.size
    crop_w, crop_h = size
    if crop_w > width or crop_h > height:
        raise ValueError("Crop size must be <= image size")
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h
    return image.crop((left, top, right, bottom))


def to_numpy(image: Image.Image, *, dtype=np.float32) -> np.ndarray:
    """Convert PIL image to numpy array."""

    return np.asarray(image, dtype=dtype)


def normalize_array(array: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Channel-wise normalization on numpy array (HWC or CHW)."""

    arr = array.astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] == len(mean):
        for c, (m, s) in enumerate(zip(mean, std)):
            arr[c] = (arr[c] - m) / s
        return arr
    if arr.ndim == 3 and arr.shape[2] == len(mean):
        for c, (m, s) in enumerate(zip(mean, std)):
            arr[..., c] = (arr[..., c] - m) / s
        return arr
    raise ValueError("Array channel dimension does not match mean/std length")


def random_horizontal_flip(image: Image.Image, *, prob: float = 0.5) -> Image.Image:
    """Randomly flip image horizontally with probability prob."""

    if prob <= 0:
        return image
    if prob >= 1:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.rand() < prob:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


class Compose:
    """Compose a sequence of preprocessing callables."""

    def __init__(self, transforms: Iterable[Callable]) -> None:
        self.transforms: List[Callable] = list(transforms)

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class ImagePreprocessor:
    """Utility to preprocess image paths into numpy arrays."""

    def __init__(self, *, resize: Optional[Tuple[int, int]] = None,
                 crop: Optional[Tuple[int, int]] = None,
                 normalize_mean: Optional[Sequence[float]] = None,
                 normalize_std: Optional[Sequence[float]] = None,
                 augmentations: Optional[Iterable[Callable[[Image.Image], Image.Image]]] = None,
                 output_tensor: bool = False) -> None:
        self.resize = resize
        self.crop = crop
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.augmentations = list(augmentations) if augmentations else []
        self.output_tensor = output_tensor

    def process(self, path: str) -> np.ndarray:
        image = load_image(path)
        for aug in self.augmentations:
            image = aug(image)
        if self.resize is not None:
            image = resize_image(image, self.resize)
        if self.crop is not None:
            image = center_crop(image, self.crop)
        array = to_numpy(image)
        if self.normalize_mean is not None and self.normalize_std is not None:
            array = normalize_array(array, self.normalize_mean, self.normalize_std)
        if self.output_tensor:
            array = np.transpose(array, (2, 0, 1))
        return array

    def batch_process(self, paths: Iterable[str]) -> np.ndarray:
        arrays = [self.process(path) for path in paths]
        return np.stack(arrays)
