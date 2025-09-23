import numpy as np

from pyimgano.utils import (
    adaptive_threshold,
    defect_preprocess_pipeline,
    enhance_edges,
    normalize_illumination,
)


def test_normalize_illumination_shapes():
    image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    normalized = normalize_illumination(image)
    assert normalized.shape == image.shape


def test_enhance_edges():
    image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    edges = enhance_edges(image)
    assert edges.shape == image.shape


def test_adaptive_threshold_binary():
    image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    thresh = adaptive_threshold(image)
    assert set(np.unique(thresh)).issubset({0, 255})


def test_default_defect_pipeline():
    image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    mask = defect_preprocess_pipeline(image)
    assert mask.shape == image.shape
