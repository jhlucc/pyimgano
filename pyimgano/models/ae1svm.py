# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from .baseCv import BaseVisionDeepDetector
from .registry import register_model

from pyod.utils.stat_models import pairwise_distances_no_broadcast
from pyod.utils.torch_utility import get_activation_by_name


class Flatten(nn.Module):
    """将 (B, C, H, W) 的图像张量展平为 (B, C*H*W) 的向量。"""
    def forward(self, x):
        return x.view(x.size(0), -1)

class Unflatten(nn.Module):
    """将向量恢复为图像张量的形状。"""
    def __init__(self, channel, height, width):
        super(Unflatten, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width
    def forward(self, x):
        return x.view(x.size(0), self.channel, self.height, self.width)

class RandomFourierFeatures(nn.Module):
    def __init__(self, input_dim, output_dim, sigma=1.0):
        super(RandomFourierFeatures, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim) * sigma)
        self.bias = nn.Parameter(torch.randn(output_dim) * 2 * np.pi)

    def forward(self, x):
        x = torch.matmul(x, self.weights) + self.bias
        return torch.cos(x)

class InnerAE1SVM(nn.Module):

    def __init__(self, n_features, encoding_dim, rff_dim, C, H, W, # 新增图像尺寸参数
                 sigma=1.0, hidden_neurons=(128, 64), dropout_rate=0.2,
                 batch_norm=True, hidden_activation='relu'):
        super(InnerAE1SVM, self).__init__()

        self.flatten = Flatten()
        self.unflatten = Unflatten(C, H, W)

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.rff = RandomFourierFeatures(encoding_dim, rff_dim, sigma)
        self.svm_weights = nn.Parameter(torch.randn(rff_dim))
        self.svm_bias = nn.Parameter(torch.randn(1))
        activation = get_activation_by_name(hidden_activation)

        # Encoder结构
        layers_neurons_encoder = [n_features, *hidden_neurons, encoding_dim]
        for idx in range(len(layers_neurons_encoder) - 1):
            self.encoder.add_module(f"linear{idx}", nn.Linear(layers_neurons_encoder[idx], layers_neurons_encoder[idx + 1]))
            if batch_norm: self.encoder.add_module(f"batch_norm{idx}", nn.BatchNorm1d(layers_neurons_encoder[idx + 1]))
            self.encoder.add_module(f"activation{idx}", activation)
            self.encoder.add_module(f"dropout{idx}", nn.Dropout(dropout_rate))

        # Decoder结构
        layers_neurons_decoder = layers_neurons_encoder[::-1]
        for idx in range(len(layers_neurons_decoder) - 1):
            self.decoder.add_module(f"linear{idx}", nn.Linear(layers_neurons_decoder[idx], layers_neurons_decoder[idx + 1]))
            if batch_norm and idx < len(layers_neurons_decoder) - 2: self.decoder.add_module(f"batch_norm{idx}", nn.BatchNorm1d(layers_neurons_decoder[idx + 1]))
            self.decoder.add_module(f"activation{idx}", activation)
            if idx < len(layers_neurons_decoder) - 2: self.decoder.add_module(f"dropout{idx}", nn.Dropout(dropout_rate))

    def forward(self, x):
        original_shape = x.shape
        x = self.flatten(x)
        encoded = self.encoder(x)
        rff_features = self.rff(encoded)
        decoded_flat = self.decoder(encoded)
        reconstructed = self.unflatten(decoded_flat)
        # 确保输出尺寸与输入一致
        if reconstructed.shape != original_shape:
            reconstructed = torch.nn.functional.interpolate(reconstructed, size=(original_shape[2], original_shape[3]), mode='bilinear', align_corners=False)
        return reconstructed, rff_features

    def svm_decision_function(self, rff_features):
        return torch.matmul(rff_features, self.svm_weights) + self.svm_bias

@register_model(
    "vision_ae1svm",
    tags=("vision", "deep", "svm"),
    metadata={"description": "自编码器 + 一类 SVM 组合的视觉检测器"},
)
class VisionAE1SVM(BaseVisionDeepDetector):
    """
    一个经过重构的、用于视觉任务的 AE1SVM 检测器。
    """
    def __init__(self, hidden_neurons=None, alpha=1.0, sigma=1.0,
                 kernel_approx_features=1000, encoding_dim=32,
                 image_shape=(3, 224, 224), # 新增：需要知道图像输入形状
                 **kwargs):

        # 存储
        self.hidden_neurons = hidden_neurons if hidden_neurons is not None else [128, 64]
        self.alpha = alpha
        self.sigma = sigma
        self.kernel_approx_features = kernel_approx_features
        self.encoding_dim = encoding_dim
        self.image_shape = image_shape # C, H, W

        # 将通用参数 (lr, epoch_num, contamination 等) 传递给父类
        super(VisionAE1SVM, self).__init__(**kwargs)

    def _build_model(self):
        """定义模型结构。"""
        n_features = self.image_shape[0] * self.image_shape[1] * self.image_shape[2]
        model = InnerAE1SVM(
            n_features=n_features,
            encoding_dim=self.encoding_dim,
            rff_dim=self.kernel_approx_features,
            C=self.image_shape[0], H=self.image_shape[1], W=self.image_shape[2],
            sigma=self.sigma,
            hidden_neurons=self.hidden_neurons,
            dropout_rate=0.2,
            batch_norm=True
        )
        return model

    def _train_loop(self, train_loader):
        """定义训练逻辑。"""
        for epoch in range(self.epoch_num):
            overall_loss = []
            for images, _ in train_loader:
                images = images.to(self.device).float()

                # 前向传播
                reconstructions, rff_features = self.model(images)

                # 计算组合损失 (这部分是 AE1SVM 的核心)
                recon_loss = self.criterion(images, reconstructions)
                svm_scores = self.model.svm_decision_function(rff_features)
                svm_loss = torch.mean(torch.clamp(1 - svm_scores, min=0))
                loss = self.alpha * recon_loss + svm_loss

                # 反向传播与优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                overall_loss.append(loss.item())

            if self.verbose > 1:
                print(f"Epoch {epoch + 1}/{self.epoch_num}, Loss: {np.mean(overall_loss):.4f}")

    def _evaluate_loop(self, data_loader):
        """【定义评估逻辑，返回异常分数。"""
        all_scores = []
        self.model.eval() # 基类会自动处理 .train() 和 .eval()，但这里写明更清晰
        with torch.no_grad():
            for images, _ in data_loader:
                images_gpu = images.to(self.device).float()
                reconstructions, _ = self.model(images_gpu)

                # 计算重构误差作为分数
                scores = pairwise_distances_no_broadcast(images.numpy().reshape(images.shape[0], -1),
                                                         reconstructions.cpu().numpy().reshape(images.shape[0], -1))
                all_scores.extend(scores)
        return all_scores
