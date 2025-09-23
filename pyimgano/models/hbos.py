# -*- coding: utf-8 -*-
"""HBOS detector wrapper for PyImgAno vision pipeline."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .baseml import BaseVisionDetector
from .registry import register_model

try:  # pragma: no cover - optional dependency guard
    from pyod.models.hbos import HBOS as _PyODHBOS
except ImportError as exc:  # pragma: no cover - surface install guidance
    _PyODHBOS = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class CoreHBOS(_PyODHBOS if _PyODHBOS is not None else object):
    """Shallow wrapper bridging PyOD HBOS to registry."""

    def __init__(self, *args, **kwargs):
        if _PyODHBOS is None:
            raise ImportError(
                "pyod.models.hbos is unavailable. Install pyod to use HBOS."
            ) from _IMPORT_ERROR
        super().__init__(*args, **kwargs)  # type: ignore[misc]


@register_model(
    "vision_hbos",
    tags=("vision", "classical"),
    metadata={"description": "Vision wrapper for histogram-based outlier detector"},
)
class VisionHBOS(BaseVisionDetector):
    def __init__(self, *, feature_extractor, contamination: float = 0.1, **kwargs):
        self.detector_kwargs = dict(contamination=contamination, **kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreHBOS(**self.detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        features = np.asarray(self.feature_extractor.extract(X))
        self.detector.fit(features)
        self.decision_scores_ = self.detector.decision_scores_
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        features = np.asarray(self.feature_extractor.extract(X))
        return self.detector.decision_function(features)
