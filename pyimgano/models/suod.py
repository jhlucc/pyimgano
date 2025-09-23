# -*- coding: utf-8 -*-
"""SUOD integration for PyImgAno."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .baseml import BaseVisionDetector
from .registry import register_model

try:  # pragma: no cover - optional dependency guard
    from pyod.models.suod import SUOD as _PyODSUOD
except ImportError as exc:  # pragma: no cover - message stored for runtime error
    _PyODSUOD = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class CoreSUOD(_PyODSUOD if _PyODSUOD is not None else object):
    """Direct wrapper on PyOD SUOD class to expose within registry."""

    def __init__(self, *args, **kwargs):
        if _PyODSUOD is None:
            raise ImportError(
                "pyod.models.suod is unavailable. Install pyod and suod package to use SUOD."
            ) from _IMPORT_ERROR
        super().__init__(*args, **kwargs)  # type: ignore[misc]


@register_model(
    "vision_suod",
    tags=("vision", "classical", "ensemble"),
    metadata={"description": "Vision wrapper for SUOD ensemble detector"},
)
class VisionSUOD(BaseVisionDetector):
    """Vision-friendly SUOD wrapper using project feature extractors."""

    def __init__(self, *, feature_extractor, contamination: float = 0.1, **kwargs):
        self.detector_kwargs = kwargs
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreSUOD(contamination=self.contamination, **self.detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        features = np.asarray(self.feature_extractor.extract(X))
        self.detector.fit(features)
        self.decision_scores_ = self.detector.decision_scores_
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        features = np.asarray(self.feature_extractor.extract(X))
        return self.detector.decision_function(features)
