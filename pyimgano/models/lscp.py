# -*- coding: utf-8 -*-
"""LSCP ensemble wrapper for PyImgAno vision pipeline."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .baseml import BaseVisionDetector
from .registry import register_model

try:  # pragma: no cover
    from pyod.models.lscp import LSCP as _PyODLSCP
except ImportError as exc:  # pragma: no cover
    _PyODLSCP = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class CoreLSCP(_PyODLSCP if _PyODLSCP is not None else object):
    def __init__(self, *args, **kwargs):
        if _PyODLSCP is None:
            raise ImportError(
                "pyod.models.lscp is unavailable. Install pyod to use LSCP."
            ) from _IMPORT_ERROR
        super().__init__(*args, **kwargs)  # type: ignore[misc]


@register_model(
    "vision_lscp",
    tags=("vision", "classical", "ensemble"),
    metadata={"description": "Vision wrapper for LSCP detector ensemble"},
)
class VisionLSCP(BaseVisionDetector):
    def __init__(self, *, feature_extractor, contamination: float = 0.1, detector_list=None, **kwargs):
        if detector_list is None:
            raise ValueError("VisionLSCP requires a 'detector_list' of base detectors.")
        self.detector_kwargs = dict(detector_list=detector_list, contamination=contamination, **kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreLSCP(**self.detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        features = np.asarray(self.feature_extractor.extract(X))
        self.detector.fit(features)
        self.decision_scores_ = self.detector.decision_scores_
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        features = np.asarray(self.feature_extractor.extract(X))
        return self.detector.decision_function(features)
