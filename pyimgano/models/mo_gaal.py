# -*- coding: utf-8 -*-
"""MO-GAAL integration within PyImgAno."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from .baseml import BaseVisionDetector
from .registry import register_model

try:  # pragma: no cover - optional dependency
    from pyod.models.mo_gaal import MO_GAAL as _PyODMO_GAAL
except ImportError as exc:  # pragma: no cover - informational message
    _PyODMO_GAAL = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class CoreMO_GAAL(_PyODMO_GAAL if _PyODMO_GAAL is not None else object):
    """Wrapper exposing PyOD's MO_GAAL when available."""

    def __init__(self, *args, **kwargs):
        if _PyODMO_GAAL is None:
            raise ImportError(
                "pyod.models.mo_gaal is unavailable. Install pyod to use MO_GAAL."
            ) from _IMPORT_ERROR
        super().__init__(*args, **kwargs)  # type: ignore[misc]


@register_model(
    "vision_mo_gaal",
    tags=("vision", "deep", "gan"),
    metadata={"description": "Vision wrapper for MO-GAAL anomaly detector"},
)
class VisionMOGAAL(BaseVisionDetector):
    """Vision-friendly wrapper around MO-GAAL."""

    def __init__(self, *, feature_extractor, contamination: float = 0.1, **kwargs):
        self.detector_kwargs = kwargs
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreMO_GAAL(contamination=self.contamination, **self.detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        features = np.asarray(self.feature_extractor.extract(X))
        self.detector.fit(features)
        self.decision_scores_ = self.detector.decision_scores_
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        features = np.asarray(self.feature_extractor.extract(X))
        return self.detector.decision_function(features)
