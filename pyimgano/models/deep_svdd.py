# -*- coding: utf-8 -*-
"""DeepSVDD 异常检测实现 (PyTorch 版本)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from torch.utils.data import DataLoader, TensorDataset

from pyod.models.base import BaseDetector

from .baseCv import BaseVisionDeepDetector
from .registry import register_model


OPTIMIZER_DICT = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
    "adadelta": optim.Adadelta,
    "adamw": optim.AdamW,
    "nadam": optim.NAdam,
    "sparseadam": optim.SparseAdam,
    "asgd": optim.ASGD,
    "lbfgs": optim.LBFGS,
}


def _get_activation(name: str) -> nn.Module:
    from pyimgano.utils.torch_utility import get_activation_by_name

    return get_activation_by_name(name)


class InnerDeepSVDD(nn.Module):
    """DeepSVDD 神经网络主体。"""

    def __init__(
        self,
        n_features: int,
        use_autoencoder: bool,
        hidden_neurons,
        hidden_activation: str,
        output_activation: str,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.use_autoencoder = use_autoencoder
        self.hidden_neurons = list(hidden_neurons or [64, 32])
        if len(self.hidden_neurons) < 2:
            raise ValueError("hidden_neurons 至少包含两个元素，以便构建输出层")
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.net = self._build_network()
        self.center = None

    # ------------------------------------------------------------------
    def _build_network(self) -> nn.Sequential:
        layers = nn.Sequential()
        layers.add_module("linear0", nn.Linear(self.n_features, self.hidden_neurons[0], bias=False))
        layers.add_module("act0", _get_activation(self.hidden_activation))

        for idx in range(1, len(self.hidden_neurons) - 1):
            layers.add_module(
                f"linear{idx}",
                nn.Linear(self.hidden_neurons[idx - 1], self.hidden_neurons[idx], bias=False),
            )
            layers.add_module(f"act{idx}", _get_activation(self.hidden_activation))
            layers.add_module(f"drop{idx}", nn.Dropout(self.dropout_rate))

        layers.add_module(
            "net_output",
            nn.Linear(self.hidden_neurons[-2], self.hidden_neurons[-1], bias=False),
        )
        layers.add_module(
            f"act{len(self.hidden_neurons) - 1}", _get_activation(self.hidden_activation)
        )

        if self.use_autoencoder:
            for idx in range(len(self.hidden_neurons) - 1, 0, -1):
                layers.add_module(
                    f"linear_d{idx}",
                    nn.Linear(self.hidden_neurons[idx], self.hidden_neurons[idx - 1], bias=False),
                )
                layers.add_module(f"act_d{idx}", _get_activation(self.hidden_activation))
                layers.add_module(f"drop_d{idx}", nn.Dropout(self.dropout_rate))

            layers.add_module("recon", nn.Linear(self.hidden_neurons[0], self.n_features, bias=False))
            layers.add_module("recon_act", _get_activation(self.output_activation))

        return layers

    @torch.no_grad()
    def init_center(self, features: torch.Tensor, eps: float = 0.1) -> None:
        self.eval()
        representation = {}
        handle = self.net._modules.get("net_output").register_forward_hook(
            lambda module, inp, out: representation.update({"net_output": out})
        )
        _ = self.forward(features)
        handle.remove()

        center = representation["net_output"].mean(dim=0)
        center = torch.where((center.abs() < eps) & (center < 0), torch.full_like(center, -eps), center)
        center = torch.where((center.abs() < eps) & (center > 0), torch.full_like(center, eps), center)
        self.center = center.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model(
    "core_deep_svdd",
    tags=("deep", "torch", "one-class"),
    metadata={"description": "核心 DeepSVDD 异常检测器"},
)
class CoreDeepSVDD(BaseDetector):
    """核心 DeepSVDD 实现，复用 PyOD BaseDetector 接口。"""

    def __init__(
        self,
        n_features: int,
        *,
        center=None,
        use_autoencoder: bool = False,
        hidden_neurons=None,
        hidden_activation: str = "relu",
        output_activation: str = "sigmoid",
        optimizer: str = "adam",
        epochs: int = 100,
        batch_size: int = 32,
        dropout_rate: float = 0.2,
        l2_weight: float = 0.1,
        validation_size: float = 0.1,
        preprocessing: bool = True,
        verbose: int = 1,
        random_state: int | None = None,
        contamination: float = 0.1,
    ) -> None:
        super().__init__(contamination=contamination)
        self.n_features = n_features
        self.center = center
        self.use_autoencoder = use_autoencoder
        self.hidden_neurons = hidden_neurons or [64, 32]
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer_name = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_weight = l2_weight
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state
        self.scaler = None
        self.model = None
        self.best_model_state = None

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        from ..utils.utility import check_parameter  # type: ignore

        check_parameter(
            dropout_rate,
            0,
            1,
            include_left=True,
            include_right=False,
            param_name="dropout_rate",
        )

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        X = check_array(X)
        self._set_n_classes(y)

        if self.preprocessing:
            self.scaler = StandardScaler()
            X_norm = self.scaler.fit_transform(X)
        else:
            X_norm = X.copy()

        indices = np.arange(X_norm.shape[0])
        np.random.shuffle(indices)
        X_norm = X_norm[indices]

        self.model = InnerDeepSVDD(
            n_features=self.n_features,
            use_autoencoder=self.use_autoencoder,
            hidden_neurons=self.hidden_neurons,
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation,
            dropout_rate=self.dropout_rate,
        )

        tensor_data = torch.tensor(X_norm, dtype=torch.float32)
        if self.center is None:
            self.model.init_center(tensor_data)
            self.center = self.model.center
        else:
            self.center = torch.tensor(self.center, dtype=torch.float32)

        dataset = TensorDataset(tensor_data, tensor_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer_cls = OPTIMIZER_DICT.get(self.optimizer_name.lower())
        if optimizer_cls is None:
            raise ValueError(f"未知优化器: {self.optimizer_name}")
        optimizer = optimizer_cls(self.model.parameters(), weight_decay=self.l2_weight)

        best_loss = float("inf")

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            self.model.train()

            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                dist = torch.sum((outputs - self.center) ** 2, dim=-1)
                if self.use_autoencoder:
                    loss = dist.mean() + torch.mean(torch.square(outputs - batch_x))
                else:
                    loss = dist.mean()

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)

            epoch_loss /= X_norm.shape[0]
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.best_model_state = self.model.state_dict()

            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss:.6f}")

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        X = check_array(X)

        if self.preprocessing and self.scaler is not None:
            X_norm = self.scaler.transform(X)
        else:
            X_norm = X.copy()

        tensor_data = torch.tensor(X_norm, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensor_data)
            dist = torch.sum((outputs - self.center) ** 2, dim=-1)

        return dist.numpy()


@register_model(
    "vision_deep_svdd",
    tags=("vision", "deep", "torch"),
    metadata={"description": "基于 DeepSVDD 的视觉异常检测器"},
)
class VisionDeepSVDD(BaseVisionDeepDetector):
    """结合 BaseVisionDeepDetector 的视觉版 DeepSVDD。"""

    def __init__(
        self,
        n_features: int,
        *,
        center=None,
        use_autoencoder: bool = False,
        hidden_neurons=None,
        hidden_activation: str = "relu",
        output_activation: str = "sigmoid",
        optimizer: str = "adam",
        epochs: int = 100,
        batch_size: int = 32,
        dropout_rate: float = 0.2,
        l2_weight: float = 0.1,
        validation_size: float = 0.1,
        preprocessing: bool = True,
        verbose: int = 1,
        random_state: int | None = None,
        contamination: float = 0.1,
        **kwargs,
    ) -> None:
        self.detector_params = dict(
            n_features=n_features,
            center=center,
            use_autoencoder=use_autoencoder,
            hidden_neurons=hidden_neurons,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            dropout_rate=dropout_rate,
            l2_weight=l2_weight,
            validation_size=validation_size,
            preprocessing=preprocessing,
            verbose=verbose,
            random_state=random_state,
            contamination=contamination,
        )

        super().__init__(
            contamination=contamination,
            preprocessing=preprocessing,
            epoch_num=epochs,
            batch_size=batch_size,
            **kwargs,
        )

    def _build_detector(self):
        return CoreDeepSVDD(**self.detector_params)
