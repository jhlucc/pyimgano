import numpy as np
import torch
from torchvision import models, transforms
from scipy.spatial.distance import mahalanobis
import cv2
import os
from sklearn.random_projection import GaussianRandomProjection

from .registry import register_model


@register_model(
    "padim",
    tags=("vision", "deep", "patch", "distribution"),
    metadata={"description": "PaDiM 风格统计异常检测"},
)
class PaDiM:
    """
    PaDiM - 轻量级异常检测，适合边缘部署
    只需要存储均值和协方差，内存占用小
    """

    def __init__(self, backbone='resnet18', d_reduced=128, device='cpu'):
        self.device = device
        self.d_reduced = d_reduced

        # 特征提取器
        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.t_d = 448  # layer2 + layer3的特征维度

        self.model.eval()
        self.model.to(device)

        # 随机投影降维
        self.random_projection = GaussianRandomProjection(
            n_components=d_reduced,
            random_state=42
        )

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 统计参数
        self.means = None
        self.inv_covs = None

    def extract_features(self, img_path):
        """提取图像特征"""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        features = {}

        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output

            return hook

        # 注册钩子获取中间层特征
        handles = []
        for name, layer in self.model.named_modules():
            if name in ['layer2', 'layer3']:
                handle = layer.register_forward_hook(hook_fn(name))
                handles.append(handle)

        with torch.no_grad():
            _ = self.model(img_tensor)

        for handle in handles:
            handle.remove()

        # 组合layer2和layer3的特征
        layer2_feat = features['layer2']  # [1, 128, 28, 28]
        layer3_feat = features['layer3']  # [1, 256, 14, 14]

        # 上采样layer3到layer2的尺寸
        layer3_feat = torch.nn.functional.interpolate(
            layer3_feat, size=layer2_feat.shape[-2:], mode='bilinear'
        )

        # 拼接特征
        features = torch.cat([layer2_feat, layer3_feat], dim=1)  # [1, 384, 28, 28]
        B, C, H, W = features.shape
        features = features.reshape(B, C, H * W).permute(0, 2, 1)  # [1, H*W, C]

        return features.squeeze(0).cpu().numpy()

    def fit(self, train_folder):
        """训练：计算正常图像的统计参数"""
        print("训练PaDiM...")

        # 首先确定降维矩阵
        temp_features = []
        for i, filename in enumerate(os.listdir(train_folder)):
            if filename.endswith('.jpg') and i < 10:  # 用前10张确定降维
                img_path = os.path.join(train_folder, filename)
                feat = self.extract_features(img_path)
                temp_features.append(feat)

        temp_features = np.vstack(temp_features)
        self.random_projection.fit(temp_features)

        # 提取所有训练图像的特征
        all_features = []
        for filename in os.listdir(train_folder):
            if filename.endswith('.jpg'):
                img_path = os.path.join(train_folder, filename)
                try:
                    features = self.extract_features(img_path)
                    # 降维
                    features_reduced = self.random_projection.transform(features)
                    all_features.append(features_reduced)
                except Exception as e:
                    print(f"处理 {filename} 出错: {e}")

        # 计算每个位置的均值和协方差
        all_features = np.array(all_features)  # [N, H*W, d_reduced]
        N, n_patches, _ = all_features.shape

        self.means = np.mean(all_features, axis=0)  # [H*W, d_reduced]

        # 计算协方差（加入正则化避免奇异）
        self.inv_covs = []
        for i in range(n_patches):
            patch_features = all_features[:, i, :]  # [N, d_reduced]
            cov = np.cov(patch_features.T) + 0.01 * np.eye(self.d_reduced)
            inv_cov = np.linalg.inv(cov)
            self.inv_covs.append(inv_cov)

        self.inv_covs = np.array(self.inv_covs)
        print(f"训练完成！统计参数形状: {self.means.shape}")

        return self

    def predict(self, img_path):
        """预测"""
        # 提取特征
        features = self.extract_features(img_path)
        features_reduced = self.random_projection.transform(features)

        # 计算马氏距离
        distances = []
        for i in range(features_reduced.shape[0]):
            feat = features_reduced[i]
            mean = self.means[i]
            inv_cov = self.inv_covs[i]

            # 马氏距离
            dist = np.sqrt((feat - mean).T @ inv_cov @ (feat - mean))
            distances.append(dist)

        distances = np.array(distances)

        # 图像级异常分数
        anomaly_score = np.max(distances)

        # 简单阈值（可根据验证集调整）
        threshold = 5.0  # 经验值
        is_anomaly = anomaly_score > threshold

        return {
            'image': os.path.basename(img_path),
            'is_normal': not is_anomaly,
            'prediction': '正常' if not is_anomaly else '异常',
            'anomaly_score': float(anomaly_score)
        }


# 使用示例
if __name__ == "__main__":
    detector = PaDiM(backbone='resnet18', d_reduced=128)
    detector.fit("/Computer/data/temp11/程序正常")
    result = detector.predict("/Computer/data/temp11/程序正常/0a2d861c87144f7b85ceda61854ffa92.jpg")
    print(result)
