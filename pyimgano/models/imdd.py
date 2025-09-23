# -*- coding: utf-8 -*-
"""IMDD (Linear Model Deviation-based Detector) vision integration."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numba import njit
from scipy import stats
from sklearn.utils import check_array, check_random_state

from pyod.utils import check_parameter

from .baseml import BaseVisionDetector
from .registry import register_model


@njit
def _aad(x):
    return np.mean(np.abs(x - np.mean(x)))


def _resolve_distance_measure(dis_measure: str):
    if dis_measure == "aad":
        return _aad
    if dis_measure == "var":
        return np.var
    if dis_measure == "iqr":
        return stats.iqr
    raise ValueError(
        "Unknown dissimilarity measure type, choose from {'aad','var','iqr'}"
    )


class CoreIMDD:
    """Pure NumPy implementation of LMDD/IMDD detector."""

    def __init__(self, contamination: float = 0.1, n_iter: int = 50, dis_measure: str = "aad",
                 random_state=None) -> None:
        self.contamination = contamination
        check_parameter(n_iter, low=1, param_name="n_iter")
        self.n_iter = n_iter
        self.random_state = check_random_state(random_state)
        self.dis_measure_name = dis_measure
        self.dis_measure = _resolve_distance_measure(dis_measure)
        self.decision_scores_ = None

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        X = check_array(X)
        self.decision_scores_ = self._compute_scores(X)
        return self

    def decision_function(self, X):
        X = check_array(X)
        return self._compute_scores(X)

    # ------------------------------------------------------------------
    def _compute_scores(self, X):
        dis = np.zeros(X.shape[0])
        card = np.zeros(X.shape[0])

        # base order computation
        base_scores = self._smoothing_factor(X)
        self._update_cardinality(base_scores, card, X.shape[0])
        dis = np.maximum(dis, base_scores)

        indices = np.arange(X.shape[0])
        rs = check_random_state(self.random_state.randint(0, 2**31 - 1))
        for _ in range(self.n_iter):
            rs.shuffle(indices)
            shuffled = X[indices]
            scores = self._smoothing_factor(shuffled)[np.argsort(indices)]
            current_card = X.shape[0] - np.count_nonzero(scores)
            better = scores > dis
            dis[better] = scores[better]
            card[better] = current_card
        return dis * card

    def _update_cardinality(self, scores, card, n_samples):
        card[:] = np.where(scores > 0, n_samples - np.count_nonzero(scores), card)

    def _smoothing_factor(self, X):
        res = np.zeros(X.shape[0])
        best_delta = -np.inf
        best_idx = 0
        for i in range(1, X.shape[0]):
            delta = self.dis_measure(X[: i + 1]) - self.dis_measure(X[:i])
            if delta > best_delta:
                best_delta = delta
                best_idx = i

        if best_delta <= 0:
            return res
        res[best_idx] = best_delta
        prefix = X[: best_idx + 1]
        prefix_prev = X[:best_idx]
        for k in range(best_idx + 1, X.shape[0]):
            diff = self.dis_measure(np.vstack((prefix_prev, X[k]))) - self.dis_measure(
                np.vstack((prefix, X[k]))
            )
            if diff >= 0:
                res[k] = diff + best_delta
        return res


@register_model(
    "vision_imdd",
    tags=("vision", "classical"),
    metadata={"description": "Vision wrapper for IMDD deviation detector"},
)
class VisionIMDD(BaseVisionDetector):
    def __init__(self, *, feature_extractor, contamination: float = 0.1, **kwargs):
        self.detector_kwargs = dict(contamination=contamination, **kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreIMDD(**self.detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        features = np.asarray(self.feature_extractor.extract(X))
        self.detector.fit(features)
        self.decision_scores_ = self.detector.decision_scores_
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        features = np.asarray(self.feature_extractor.extract(X))
        return self.detector.decision_function(features)
