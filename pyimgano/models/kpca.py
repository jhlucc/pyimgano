# -*- coding: utf-8 -*-
"""Kernel PCA 异常检测器实现。"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from pyod.models.base import BaseDetector

from .baseml import BaseVisionDetector
from .registry import register_model


def _check_parameter(param, low=None, high=None, *, include_left=True,
                     include_right=True, name="parameter") -> None:
    """参数范围校验，保持与 PyOD 版本一致的行为。"""

    if low is not None:
        if include_left:
            if param < low:
                raise ValueError(f"{name}={param} 必须 >= {low}")
        else:
            if param <= low:
                raise ValueError(f"{name}={param} 必须 > {low}")

    if high is not None:
        if include_right:
            if param > high:
                raise ValueError(f"{name}={param} 必须 <= {high}")
        else:
            if param >= high:
                raise ValueError(f"{name}={param} 必须 < {high}")


class _PyODKernelPCA(KernelPCA):
    """轻量包装 sklearn KernelPCA 以暴露内部属性。"""

    def __init__(
        self,
        *,
        n_components=None,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1.0,
        fit_inverse_transform=False,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        remove_zero_eig=False,
        copy_X=True,
        n_jobs=None,
        random_state=None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            alpha=alpha,
            fit_inverse_transform=fit_inverse_transform,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            remove_zero_eig=remove_zero_eig,
            copy_X=copy_X,
            n_jobs=n_jobs,
            random_state=check_random_state(random_state),
        )

    @property
    def centerer(self):
        return self._centerer

    @property
    def kernel_callable(self):
        return self._get_kernel


@register_model(
    "core_kpca",
    tags=("classical", "kernel", "projection"),
    metadata={"description": "核心 Kernel PCA 异常检测器"},
)
class CoreKPCA(BaseDetector):
    """Kernel PCA 异常检测器实现。"""

    def __init__(
        self,
        contamination=0.1,
        *,
        n_components=None,
        n_selected_components=None,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1.0,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        remove_zero_eig=False,
        copy_X=True,
        n_jobs=None,
        sampling=False,
        subset_size=20,
        random_state=None,
    ) -> None:
        super().__init__(contamination=contamination)
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.alpha = alpha
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.remove_zero_eig = remove_zero_eig
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.sampling = sampling
        self.subset_size = subset_size
        self.random_state = check_random_state(random_state)
        self.n_selected_components_ = None
        self.kpca = None

    # ------------------------------------------------------------------
    def _check_subset_size(self, array: np.ndarray) -> int:
        n_samples = array.shape[0]

        if isinstance(self.subset_size, int):
            if 0 < self.subset_size <= n_samples:
                return self.subset_size
            raise ValueError(
                f"subset_size={self.subset_size} 必须位于 (0, {n_samples}] 内"
            )

        if isinstance(self.subset_size, float):
            if 0.0 < self.subset_size <= 1.0:
                return max(1, int(self.subset_size * n_samples))
            raise ValueError("subset_size 为浮点数时需位于 (0.0, 1.0]")

        raise TypeError("subset_size 仅支持 int 或 float")

    def fit(self, X, y=None):
        X = check_array(X, copy=self.copy_X)
        self._set_n_classes(y)

        if self.sampling:
            subset_size = self._check_subset_size(X)
            indices = self.random_state.choice(X.shape[0], size=subset_size, replace=False)
            X = X[indices, :]

        if self.n_components is None:
            n_components = X.shape[0]
        else:
            if self.n_components < 1:
                raise ValueError("n_components 应 >= 1")
            n_components = min(X.shape[0], self.n_components)

        if self.n_selected_components is None:
            self.n_selected_components_ = n_components
        else:
            self.n_selected_components_ = self.n_selected_components

        _check_parameter(
            self.n_selected_components_,
            low=1,
            high=n_components,
            include_left=True,
            include_right=True,
            name="n_selected_components",
        )

        self.kpca = _PyODKernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            kernel_params=self.kernel_params,
            alpha=self.alpha,
            fit_inverse_transform=False,
            eigen_solver=self.eigen_solver,
            tol=self.tol,
            max_iter=self.max_iter,
            remove_zero_eig=self.remove_zero_eig,
            copy_X=self.copy_X,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        transformed = self.kpca.fit_transform(X)
        transformed = transformed[:, : self.n_selected_components_]

        centerer = self.kpca.centerer
        kernel = self.kpca.kernel_callable

        potential = []
        for sample in X:
            potential.append(kernel(sample.reshape(1, -1)))
        potential = np.asarray(potential).squeeze()
        potential = potential - 2 * centerer.K_fit_rows_ + centerer.K_fit_all_

        self.decision_scores_ = potential - np.sum(np.square(transformed), axis=1)
        self._process_decision_scores()

        return self

    def decision_function(self, X):
        check_is_fitted(self, ["decision_scores_", "threshold_", "labels_"])
        X = check_array(X)

        kernel = self.kpca.kernel_callable
        centerer = self.kpca.centerer

        gram_matrix = kernel(X, self.kpca.X_fit_)
        transformed = self.kpca.transform(X)
        transformed = transformed[:, : self.n_selected_components_]

        potential = []
        for sample in X:
            potential.append(kernel(sample.reshape(1, -1)))
        potential = np.asarray(potential).squeeze()

        gram_fit_rows = np.sum(gram_matrix, axis=1) / gram_matrix.shape[1]
        potential = potential - 2 * gram_fit_rows + centerer.K_fit_all_

        return potential - np.sum(np.square(transformed), axis=1)


@register_model(
    "vision_kpca",
    tags=("vision", "classical", "kernel"),
    metadata={"description": "基于 Kernel PCA 的视觉异常检测器"},
)
class VisionKPCA(BaseVisionDetector):
    """结合特征提取器的视觉版 Kernel PCA。"""

    def __init__(
        self,
        contamination=0.1,
        feature_extractor=None,
        *,
        n_components=None,
        n_selected_components=None,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1.0,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        remove_zero_eig=False,
        copy_X=True,
        n_jobs=None,
        sampling=False,
        subset_size=20,
        random_state=None,
    ) -> None:
        self.detector_params = dict(
            contamination=contamination,
            n_components=n_components,
            n_selected_components=n_selected_components,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            alpha=alpha,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            remove_zero_eig=remove_zero_eig,
            copy_X=copy_X,
            n_jobs=n_jobs,
            sampling=sampling,
            subset_size=subset_size,
            random_state=random_state,
        )

        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreKPCA(**self.detector_params)
