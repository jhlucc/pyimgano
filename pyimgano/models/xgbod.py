# -*- coding: utf-8 -*-
"""XGBOD detector integration for PyImgAno."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .baseml import BaseVisionDetector
from .registry import register_model

try:  # pragma: no cover - optional dependency check
    from pyod.models.xgbod import XGBOD as _PyODXGBOD
except ImportError as exc:  # pragma: no cover - explicit message for users
    _PyODXGBOD = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class CoreXGBOD(_PyODXGBOD if _PyODXGBOD is not None else object):
    """Thin wrapper around PyOD's XGBOD to expose in factory registry."""

    def __init__(self, *args, **kwargs):
        if _PyODXGBOD is None:
            raise ImportError(
                "pyod.models.xgbod is unavailable. Install pyod and xgboost to use XGBOD."
            ) from _IMPORT_ERROR
        super().__init__(*args, **kwargs)  # type: ignore[misc]


@register_model(
    "vision_xgbod",
    tags=("vision", "classical", "supervised"),
    metadata={"description": "Vision wrapper for XGBOD semi-supervised detector"},
)
class VisionXGBOD(BaseVisionDetector):
    """Vision pipeline integration for XGBOD.

    Requires supervision (labels) during fitting. Input `X` should be image
    paths compatible with the provided feature extractor, and `y` a binary
    numpy array where 1 denotes anomalies.
    """

    def __init__(self, *, feature_extractor, contamination: float = 0.1, **kwargs):
        self.detector_kwargs = kwargs
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreXGBOD(**self.detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        if y is None:
            raise ValueError("VisionXGBOD.fit requires ground truth labels 'y'.")
        features = self.feature_extractor.extract(X)
        features = np.asarray(features)
        y = np.asarray(y)
        self.detector.fit(features, y)
        self.decision_scores_ = self.detector.decision_scores_
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        features = np.asarray(self.feature_extractor.extract(X))
        return self.detector.decision_function(features)
