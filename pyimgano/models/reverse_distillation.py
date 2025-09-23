# -*- coding: utf-8 -*-
"""Reverse Distillation (student-teacher) anomaly detector."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .baseCv import BaseVisionDeepDetector
from .registry import register_model
from .fastflow import ResNetFeatureExtractor


class StudentResNetExtractor(nn.Module):
    """Trainable student network mirroring selected ResNet layers."""

    def __init__(self, layers: Sequence[str] = ("layer2", "layer3", "layer4")) -> None:
        super().__init__()
        net = models.resnet18(weights=None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.selected_layers = tuple(layers)

    def forward(self, x: torch.Tensor):
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


@register_model(
    "vision_reverse_distillation",
    tags=("vision", "deep", "distillation"),
    metadata={"description": "Reverse distillation anomaly detector"},
)
class ReverseDistillation(BaseVisionDeepDetector):
    """Reverse Distillation anomaly detector (RD4AD style)."""

    def __init__(
        self,
        contamination: float = 0.1,
        *,
        backbone: str = "resnet18",
        pretrained_backbone: bool = True,
        selected_layers: Sequence[str] = ("layer2", "layer3", "layer4"),
        lr: float = 1e-3,
        epoch_num: int = 20,
        batch_size: int = 16,
        device: str | None = None,
        verbose: int = 1,
        random_state: int = 42,
    ) -> None:
        self.backbone = backbone
        self.pretrained_backbone = pretrained_backbone
        self.selected_layers = tuple(selected_layers)
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

    def build_model(self):
        self.teacher = ResNetFeatureExtractor(
            backbone=self.backbone,
            pretrained=self.pretrained_backbone,
            layers=self.selected_layers,
        ).to(self.device)
        self.student = StudentResNetExtractor(self.selected_layers).to(self.device)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
        return nn.ModuleList([self.student])

    def _forward_features(self, images: torch.Tensor):
        with torch.no_grad():
            teacher_feats = self.teacher(images)
        student_feats = self.student(images)
        return teacher_feats, student_feats

    def training_forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        images, _ = batch
        images = images.to(self.device)

        teacher_feats, student_feats = self._forward_features(images)

        loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        for t_feat, s_feat in zip(teacher_feats, student_feats):
            loss = loss + F.mse_loss(s_feat, t_feat.detach())
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def evaluating_forward(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        images, _ = batch
        images = images.to(self.device)
        teacher_feats, student_feats = self._forward_features(images)
        scores = []
        for t_feat, s_feat in zip(teacher_feats, student_feats):
            diff = (s_feat - t_feat) ** 2
            scores.append(diff.flatten(1).mean(dim=1))
        total = torch.stack(scores, dim=1).mean(dim=1)
        return total.cpu().numpy()
