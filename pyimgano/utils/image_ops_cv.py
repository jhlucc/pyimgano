"""OpenCV-based image preprocessing utilities."""

from __future__ import annotations

import math
import random
from typing import Callable, Iterable, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Basic conversions & filters
# ---------------------------------------------------------------------------


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB/BGR image to grayscale."""

    if image.ndim == 2:
        return image
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Unsupported channel format for grayscale conversion")


def gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), sigma: float = 0) -> np.ndarray:
    """Apply Gaussian blur smoothing."""

    return cv2.GaussianBlur(image, kernel_size, sigma)


def sharpen(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Sharpen image using unsharp masking."""

    blurred = gaussian_blur(image, (0, 0), sigma=1.0)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def to_gray_equalized(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale and apply histogram equalization."""

    gray = to_gray(image)
    return cv2.equalizeHist(gray)


def clahe_equalization(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE to grayscale or color images."""

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if image.ndim == 2:
        return clahe.apply(image)
    channels = cv2.split(image)
    eq_channels = [clahe.apply(to_gray(ch) if ch.ndim == 3 else ch) for ch in channels]
    return cv2.merge(eq_channels)


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 10.0) -> np.ndarray:
    """Add Gaussian noise to image."""

    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noised = image.astype(np.float32) + noise
    return np.clip(noised, 0, 255).astype(image.dtype)


def adjust_brightness_contrast(image: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """Adjust brightness and contrast (alpha: contrast, beta: brightness)."""

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def motion_blur(image: np.ndarray, kernel_size: int = 9, angle: float = 0.0) -> np.ndarray:
    """Apply motion blur with a given angle."""

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    rot_matrix = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_matrix, (kernel_size, kernel_size))
    kernel = kernel / np.sum(kernel)
    return cv2.filter2D(image, -1, kernel)


# ---------------------------------------------------------------------------
# Edge & contour operations
# ---------------------------------------------------------------------------


def canny_edges(image: np.ndarray, threshold1: float = 100, threshold2: float = 200,
                aperture_size: int = 3) -> np.ndarray:
    """Apply Canny edge detector."""

    gray = to_gray(image)
    return cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture_size)


def sobel_edges(image: np.ndarray, dx: int = 1, dy: int = 0, ksize: int = 3) -> np.ndarray:
    """Compute Sobel gradients."""

    gray = to_gray(image)
    grad = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
    return cv2.convertScaleAbs(grad)


def laplacian_edges(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Compute Laplacian edge map."""

    gray = to_gray(image)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(lap)


def find_contours(image: np.ndarray, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
    """Find contours on binary or grayscale image."""

    if image.ndim == 3:
        image = to_gray(image)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, mode, method)
    return contours, hierarchy


# ---------------------------------------------------------------------------
# Morphology operations
# ---------------------------------------------------------------------------


def erode(image: np.ndarray, kernel_size: Tuple[int, int] = (3, 3), iterations: int = 1) -> np.ndarray:
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)


def dilate(image: np.ndarray, kernel_size: Tuple[int, int] = (3, 3), iterations: int = 1) -> np.ndarray:
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)


def morphological_open(image: np.ndarray, kernel_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def morphological_close(image: np.ndarray, kernel_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------


def random_rotation(image: np.ndarray, angle_range: Tuple[float, float] = (-10, 10),
                    scale: float = 1.0) -> np.ndarray:
    angle = random.uniform(*angle_range)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def random_crop_and_resize(image: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.0),
                           output_size: Tuple[int, int] | None = None) -> np.ndarray:
    h, w = image.shape[:2]
    scale = random.uniform(*scale_range)
    crop_h, crop_w = int(h * scale), int(w * scale)
    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)
    crop = image[top:top + crop_h, left:left + crop_w]
    if output_size is not None:
        return cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)
    return crop


def random_brightness_contrast(image: np.ndarray, alpha_range: Tuple[float, float] = (0.8, 1.2),
                               beta_range: Tuple[float, float] = (-20, 20)) -> np.ndarray:
    alpha = random.uniform(*alpha_range)
    beta = random.uniform(*beta_range)
    return adjust_brightness_contrast(image, alpha, beta)


def random_gaussian_noise(image: np.ndarray, sigma_range: Tuple[float, float] = (0, 15)) -> np.ndarray:
    sigma = random.uniform(*sigma_range)
    if sigma <= 0:
        return image
    return add_gaussian_noise(image, sigma=sigma)


def apply_augmentations(image: np.ndarray, augmentations: Iterable[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    result = image
    for aug in augmentations:
        result = aug(result)
    return result


class AugmentationPipeline:
    """Chain OpenCV-based augmentation functions."""

    def __init__(self, augmentations: Iterable[Callable[[np.ndarray], np.ndarray]]) -> None:
        self.augmentations = list(augmentations)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return apply_augmentations(image, self.augmentations)
