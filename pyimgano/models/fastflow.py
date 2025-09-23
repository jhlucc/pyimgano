# -*- coding: utf-8 -*-
"""FastFlow-based visual anomaly detector implementation."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from .baseCv import BaseVisionDeepDetector
from .registry import register_model


# ---------------------------------------------------------------------------
# Flow building blocks
# ---------------------------------------------------------------------------


class ActNorm2d(nn.Module):
    """Activation normalization with data-dependent initialization."""

    def __init__(self, num_features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.initialized = False
        self.eps = eps

    def _initialize(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = x.std(dim=[0, 2, 3], keepdim=True) + self.eps
            self.bias.data.copy_(-mean)
            self.log_scale.data.copy_(torch.log(1.0 / std))
        self.initialized = True

    def forward(self, x: torch.Tensor, logdet: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            self._initialize(x)
        if logdet is None:
            logdet = x.new_zeros(x.size(0))

        H, W = x.shape[2], x.shape[3]
        if reverse:
            x = (x - self.bias) * torch.exp(-self.log_scale)
            logdet = logdet - torch.sum(self.log_scale) * H * W
        else:
            x = (x + self.bias) * torch.exp(self.log_scale)
            logdet = logdet + torch.sum(self.log_scale) * H * W
        return x, logdet


class InvConv2d(nn.Module):
    """Invertible 1x1 convolution following Glow."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        weight = torch.qr(torch.randn(num_features, num_features))[0]
        weight = weight.view(num_features, num_features, 1, 1)
        self.weight = nn.Parameter(weight)

    def _log_det(self) -> torch.Tensor:
        w = self.weight.squeeze(-1).squeeze(-1)
        return torch.logdet(w)

    def forward(self, x: torch.Tensor, logdet: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if logdet is None:
            logdet = x.new_zeros(x.size(0))
        H, W = x.shape[2], x.shape[3]
        log_abs_det = self._log_det() * H * W
        if reverse:
            weight = torch.inverse(self.weight.squeeze(-1).squeeze(-1))
            weight = weight.view_as(self.weight)
            x = F.conv2d(x, weight)
            logdet = logdet - log_abs_det
        else:
            x = F.conv2d(x, self.weight)
            logdet = logdet + log_abs_det
        return x, logdet


class AffineCoupling(nn.Module):
    """Affine coupling layer."""

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Conv2d(in_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, logdet: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if logdet is None:
            logdet = x.new_zeros(x.size(0))
        x1, x2 = torch.chunk(x, 2, dim=1)
        h = self.hidden(x1)
        shift, scale = torch.chunk(h, 2, dim=1)
        scale = torch.tanh(scale)
        if reverse:
            x2 = (x2 * torch.exp(-scale)) - shift
            logdet = logdet - scale.view(scale.size(0), -1).sum(dim=1)
        else:
            x2 = (x2 + shift) * torch.exp(scale)
            logdet = logdet + scale.view(scale.size(0), -1).sum(dim=1)
        return torch.cat([x1, x2], dim=1), logdet


class FlowStep(nn.Module):
    def __init__(self, channels: int, hidden_ratio: float = 1.5) -> None:
        super().__init__()
        hidden_channels = int(math.ceil(channels * hidden_ratio))
        if channels % 2 != 0:
            raise ValueError("FlowStep channels must be even.")
        self.actnorm = ActNorm2d(channels)
        self.invconv = InvConv2d(channels)
        self.coupling = AffineCoupling(channels, hidden_channels)

    def forward(self, x: torch.Tensor, logdet: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x, logdet = self.actnorm(x, logdet, reverse=reverse)
        x, logdet = self.invconv(x, logdet, reverse=reverse)
        x, logdet = self.coupling(x, logdet, reverse=reverse)
        return x, logdet


class FlowStage(nn.Module):
    """A sequence of FlowSteps applied to a feature map."""

    def __init__(self, channels: int, n_steps: int, hidden_ratio: float) -> None:
        super().__init__()
        self.steps = nn.ModuleList([FlowStep(channels, hidden_ratio=hidden_ratio) for _ in range(n_steps)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet = x.new_zeros(x.size(0))
        for step in self.steps:
            x, logdet = step(x, logdet, reverse=False)
        return x, logdet

    @torch.no_grad()
    def forward_no_grad(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet = x.new_zeros(x.size(0))
        for step in self.steps:
            x, logdet = step(x, logdet, reverse=False)
        return x, logdet


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


class ResNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        layers: Sequence[str] = ("layer2", "layer3", "layer4"),
    ) -> None:
        super().__init__()
        if backbone != "resnet18":
            raise ValueError("Currently only resnet18 backbone is supported.")
        weights = None
        if pretrained:
            try:  # torchvision>=0.13
                weights = models.ResNet18_Weights.DEFAULT
            except AttributeError:  # fallback older versions
                weights = models.ResNet18_Weights.IMAGENET1K_V1 if hasattr(models, "ResNet18_Weights") else "DEFAULT"
        net = models.resnet18(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.selected_layers = tuple(layers)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        feature_map = {
            "layer2": feat2,
            "layer3": feat3,
            "layer4": feat4,
        }
        return [feature_map[name] for name in self.selected_layers]


# ---------------------------------------------------------------------------
# FastFlow detector
# ---------------------------------------------------------------------------


@register_model(
    "vision_fastflow",
    tags=("vision", "deep", "flow"),
    metadata={"description": "FastFlow-based visual anomaly detector"},
)
class FastFlow(BaseVisionDeepDetector):
    """Implementation of FastFlow (ICCV'21) anomaly detector."""

    def __init__(
        self,
        contamination: float = 0.1,
        *,
        backbone: str = "resnet18",
        pretrained_backbone: bool = True,
        selected_layers: Sequence[str] = ("layer2", "layer3", "layer4"),
        embedding_dim: int = 256,
        n_flow_steps: int = 8,
        flow_hidden_ratio: float = 1.5,
        lr: float = 1e-4,
        epoch_num: int = 20,
        batch_size: int = 16,
        device: str | None = None,
        verbose: int = 1,
        random_state: int = 42,
    ) -> None:
        self.backbone = backbone
        self.pretrained_backbone = pretrained_backbone
        self.selected_layers = tuple(selected_layers)
        self.embedding_dim = embedding_dim
        if self.embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for affine coupling.")
        self.n_flow_steps = n_flow_steps
        self.flow_hidden_ratio = flow_hidden_ratio
        super().__init__(
            contamination=contamination,
            preprocessing=True,
            lr=lr,
            epoch_num=epoch_num,
            batch_size=batch_size,
            optimizer_name="adam",
            criterion_name="mse",
            device=device,
            random_state=random_state,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    def build_model(self):
        self.feature_extractor = ResNetFeatureExtractor(
            backbone=self.backbone,
            pretrained=self.pretrained_backbone,
            layers=self.selected_layers,
        ).to(self.device)
        self.feature_extractor.eval()

        adaptor_list = []
        stage_list = []
        channel_map = {"layer2": 128, "layer3": 256, "layer4": 512}
        for layer in self.selected_layers:
            in_channels = channel_map.get(layer)
            if in_channels is None:
                raise ValueError(f"Unsupported layer {layer}")
            adaptor = nn.Conv2d(in_channels, self.embedding_dim, kernel_size=1, stride=1, bias=True)
            adaptor_list.append(adaptor)
            stage_list.append(FlowStage(self.embedding_dim, self.n_flow_steps, self.flow_hidden_ratio))

        self.adapters = nn.ModuleList(adaptor_list).to(self.device)
        self.flow_stages = nn.ModuleList(stage_list).to(self.device)

        params = list(self.adapters.parameters()) + list(self.flow_stages.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

        return nn.ModuleList([self.adapters, self.flow_stages])

    # ------------------------------------------------------------------
    def _extract_features(self, images: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            feats = self.feature_extractor(images)
        return [feat.detach() for feat in feats]

    # ------------------------------------------------------------------
    def _flow_nll(self, z: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
        flat = z.view(z.size(0), -1)
        n_dims = flat.size(1)
        log_prob = (-0.5 * flat.pow(2).sum(dim=1) + logdet) / n_dims
        return -log_prob  # negative log likelihood per sample

    # ------------------------------------------------------------------
    def training_forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        images, _ = batch
        images = images.to(self.device)

        features = self._extract_features(images)

        self.adapters.train()
        self.flow_stages.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss = 0.0
        for feat, adaptor, flow in zip(features, self.adapters, self.flow_stages):
            feat = adaptor(feat.to(self.device))
            z, logdet = flow(feat)
            loss_stage = self._flow_nll(z, logdet).mean()
            loss = loss + loss_stage

        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluating_forward(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        images, _ = batch
        images = images.to(self.device)
        features = self._extract_features(images)

        self.adapters.eval()
        self.flow_stages.eval()
        scores = []
        for feat, adaptor, flow in zip(features, self.adapters, self.flow_stages):
            feat = adaptor(feat.to(self.device))
            z, logdet = flow.forward_no_grad(feat)
            stage_score = self._flow_nll(z, logdet)
            scores.append(stage_score)
        total = torch.stack(scores, dim=1).mean(dim=1)
        return total.cpu().numpy()

    # ------------------------------------------------------------------
    def fit(self, X: Iterable[str], y: Iterable[int] | None = None):
        # Override to ensure feature extractor is on correct device before DataLoader loop
        return super().fit(X, y)

    def build_model_loader(self, X: Sequence[str]) -> DataLoader:
        # Not overriding base behaviour; placeholder for compatibility.
        return super().build_model_loader(X)
