"""Specialized image preprocessing functions for defect detection."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np


def normalize_illumination(image: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    """Normalize uneven illumination using background subtraction."""

    if kernel_size % 2 == 0:
        kernel_size += 1
    blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    normalized = cv2.addWeighted(image, 1.0, blur, -1.0, 128)
    return np.clip(normalized, 0, 255).astype(image.dtype)


def background_subtraction(image: np.ndarray, kernel_size: int = 25) -> np.ndarray:
    """Remove slow-varying background via median filtering."""

    background = cv2.medianBlur(image, kernel_size)
    residual = cv2.subtract(image, background)
    return residual


def adaptive_threshold(image: np.ndarray, block_size: int = 35, c: int = 5,
                       method: str = "gaussian") -> np.ndarray:
    """Adaptive threshold suited for defect segmentation."""

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if block_size % 2 == 0:
        block_size += 1
    adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method.lower() == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    thresh = cv2.adaptiveThreshold(gray, 255, adaptive_method, cv2.THRESH_BINARY, block_size, c)
    return thresh


def top_hat(image: np.ndarray, kernel_size: Tuple[int, int] = (15, 15)) -> np.ndarray:
    """Highlight bright defects on dark background via top-hat transform."""

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def bottom_hat(image: np.ndarray, kernel_size: Tuple[int, int] = (15, 15)) -> np.ndarray:
    """Highlight dark defects on bright background via bottom-hat transform."""

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)


def difference_of_gaussian(image: np.ndarray, sigma_small: float = 1.0, sigma_large: float = 3.0) -> np.ndarray:
    """Edge enhancement using difference of Gaussian filters."""

    blur_small = cv2.GaussianBlur(image, (0, 0), sigma_small)
    blur_large = cv2.GaussianBlur(image, (0, 0), sigma_large)
    diff = cv2.subtract(blur_small, blur_large)
    return diff


def gabor_filter_bank(image: np.ndarray, frequencies: Sequence[float] = (0.1, 0.2, 0.3),
                      thetas: Sequence[float] = (0, 45, 90, 135)) -> np.ndarray:
    """Extract texture responses using a bank of Gabor filters."""

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    responses = []
    for theta in thetas:
        theta_rad = theta / 180 * math.pi
        for freq in frequencies:
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta_rad, 1 / freq, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            responses.append(np.abs(filtered))
    stacked = np.stack(responses, axis=0)
    return np.mean(stacked, axis=0)


def enhance_edges(image: np.ndarray, weight: float = 1.5) -> np.ndarray:
    """Enhance fine defects by combining Laplacian with original image."""

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    enhanced = cv2.addWeighted(gray.astype(np.float32), 1.0, lap, weight, 0)
    return np.clip(enhanced, 0, 255).astype(gray.dtype)


def local_binary_pattern(image: np.ndarray, radius: int = 1, n_points: int | None = None) -> np.ndarray:
    """Compute Local Binary Pattern for texture anomaly detection."""

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    n_points = n_points or 8 * radius
    rows, cols = gray.shape
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for y in range(radius, rows - radius):
        for x in range(radius, cols - radius):
            center = gray[y, x]
            binary = 0
            for idx, angle in enumerate(np.linspace(0, 2 * np.pi, n_points, endpoint=False)):
                yy = int(round(y + radius * math.sin(angle)))
                xx = int(round(x + radius * math.cos(angle)))
                binary |= (1 << idx) if gray[yy, xx] > center else 0
            lbp[y, x] = binary
    return lbp


def multi_scale_defect_map(image: np.ndarray, scales: Iterable[int] = (3, 5, 7)) -> np.ndarray:
    """Combine top-hat responses from multi-scale kernels."""

    responses = []
    for s in scales:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (s, s))
        responses.append(cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel))
    stacked = np.stack(responses, axis=0)
    return np.max(stacked, axis=0)


def defect_preprocess_pipeline(image: np.ndarray) -> np.ndarray:
    """Default pipeline tailored for surface defect highlighting."""

    img = normalize_illumination(image)
    img = enhance_edges(img)
    mask = multi_scale_defect_map(img)
    return mask
