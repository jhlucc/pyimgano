# -*- coding: utf-8 -*-
"""
基于ABOD算法的视觉异常检测器 (VisionABOD)。
"""
# 原始ABOD算法作者: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

#视觉基类
from .baseml import BaseVisionDetector
from .registry import register_model
# 从PyOD导入ABOD算法
from pyod.models.abod import ABOD


@register_model(
    "vision_abod",
    tags=("vision", "classical"),
    metadata={"description": "基于 ABOD 的视觉异常检测器"},
)
class VisionABOD(BaseVisionDetector):
    """
    一个基于ABOD (Angle-based Outlier Detector) 算法的视觉异常检测器。
    这个类继承了 BaseVisionDetector 的所有特色功能，包括：
    - 插件化的特征提取器。
    - 自动化的 “图像 -> 特征 -> 分数” 工作流。
    - 继承自 pyod.BaseDetector 的完整分数处理和预测接口。

    Parameters
    ----------
    contamination : float, 范围在 (0., 0.5) 之间, 可选 (默认为 0.1)
        数据集的污染程度。

    feature_extractor : object
        一个实现了 .extract(X) 方法的对象，负责将图像转换为特征向量。

    n_neighbors : int, 可选 (默认为 10)
        用于ABOD算法的邻居数量。
    method: str, 可选 (默认为'fast')
        ABOD算法使用的方法 ('fast' 或 'default')。
    """
    def __init__(self, contamination=0.1, feature_extractor=None,
                 n_neighbors=10, method='fast'):
        self.n_neighbors = n_neighbors
        self.method = method

        super(VisionABOD, self).__init__(
            contamination=contamination,
            feature_extractor=feature_extractor
        )

    def _build_detector(self):
        return ABOD(contamination=self.contamination,
                    n_neighbors=self.n_neighbors,
                    method=self.method)

# ===================================================================
#                          使 用 示 例
# ===================================================================
# from your_vision_package.models import VisionABOD
# from your_vision_package.utils import ResNetFeatureExtractor # 假设您提供了一个默认的特征提取器

# # 1. 创建一个特征提取器实例
# resnet_extractor = ResNetFeatureExtractor(device='cuda')

# # 2. 创建您的 VisionABOD 检测器，把特征提取器“插”进去
# abod_detector = VisionABOD(feature_extractor=resnet_extractor, n_neighbors=30)

# # 3. 直接用图片路径进行训练！
# # 基类会自动处理：图片 -> 特征 -> 训练ABOD
# abod_detector.fit(normal_image_paths)

# # 4. 直接用图片路径进行预测！
# # 基类会自动处理：图片 -> 特征 -> ABOD打分
# scores = abod_detector.decision_function(test_image_paths)
# labels = abod_detector.predict(test_image_paths)

# print("异常分数:", scores)
# print("预测标签:", labels)
