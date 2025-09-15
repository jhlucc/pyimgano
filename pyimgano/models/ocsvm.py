# -*- coding: utf-8 -*-
"""
通用视觉OCSVM异常检测框架
一个模块化、可扩展的视觉异常检测解决方案
支持多种特征提取器和灵活的配置
"""

import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any, Optional
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyod.models.ocsvm import OCSVM
from baseml import BaseVisionDetector


class BaseFeatureExtractor(ABC):
    """
    特征提取器的抽象基类
    所有特征提取器都必须实现extract方法
    """

    @abstractmethod
    def extract(self, image_paths: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        提取特征的核心方法

        Parameters
        ----------
        image_paths : List[str] or np.ndarray
            图像路径列表或图像数组

        Returns
        -------
        features : np.ndarray
            提取的特征矩阵 (n_samples, n_features)
        """
        pass

    def get_feature_names(self) -> List[str]:
        """获取特征名称（可选实现）"""
        return []


class PixelFeatureExtractor(BaseFeatureExtractor):
    """
    像素级特征提取器
    直接使用原始像素值或简单统计特征
    """

    def __init__(self, resize_dim=(64, 64), flatten=True, grayscale=True):
        """
        Parameters
        ----------
        resize_dim : tuple
            调整图像大小
        flatten : bool
            是否展平为一维向量
        grayscale : bool
            是否转换为灰度图
        """
        self.resize_dim = resize_dim
        self.flatten = flatten
        self.grayscale = grayscale

    def extract(self, image_paths: Union[List[str], np.ndarray]) -> np.ndarray:
        features = []

        for path in tqdm(image_paths, desc="提取像素特征"):
            try:
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"无法读取图像: {path}")

                # 调整大小
                img = cv2.resize(img, self.resize_dim)

                # 转换为灰度图（如果需要）
                if self.grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # 展平（如果需要）
                if self.flatten:
                    img = img.flatten()

                # 归一化到[0, 1]
                img = img.astype(np.float32) / 255.0

                features.append(img)

            except Exception as e:
                print(f"处理 {path} 出错: {e}")
                # 添加零特征作为占位
                if self.flatten:
                    features.append(np.zeros(np.prod(self.resize_dim)))
                else:
                    features.append(np.zeros(self.resize_dim))

        return np.array(features)


class HistogramFeatureExtractor(BaseFeatureExtractor):
    """
    直方图特征提取器
    提取颜色直方图、梯度直方图等统计特征
    """

    def __init__(self, bins=32, color_space='RGB', use_gradient=True):
        """
        Parameters
        ----------
        bins : int
            直方图的bin数量
        color_space : str
            颜色空间 ('RGB', 'HSV', 'LAB')
        use_gradient : bool
            是否包含梯度直方图
        """
        self.bins = bins
        self.color_space = color_space
        self.use_gradient = use_gradient

    def extract(self, image_paths: Union[List[str], np.ndarray]) -> np.ndarray:
        features = []

        for path in tqdm(image_paths, desc="提取直方图特征"):
            try:
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"无法读取图像: {path}")

                feature_vec = []

                # 颜色直方图
                if self.color_space == 'HSV':
                    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                elif self.color_space == 'LAB':
                    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                else:
                    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                for i in range(3):
                    hist = cv2.calcHist([img_color], [i], None, [self.bins], [0, 256])
                    hist = hist.flatten() / hist.sum()  # 归一化
                    feature_vec.extend(hist)

                # 梯度直方图
                if self.use_gradient:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    magnitude = np.sqrt(gx ** 2 + gy ** 2)
                    angle = np.arctan2(gy, gx)

                    # 梯度幅值直方图
                    mag_hist = np.histogram(magnitude, bins=self.bins)[0]
                    mag_hist = mag_hist / (mag_hist.sum() + 1e-6)
                    feature_vec.extend(mag_hist)

                    # 梯度方向直方图
                    angle_hist = np.histogram(angle, bins=self.bins, range=(-np.pi, np.pi))[0]
                    angle_hist = angle_hist / (angle_hist.sum() + 1e-6)
                    feature_vec.extend(angle_hist)

                features.append(feature_vec)

            except Exception as e:
                print(f"处理 {path} 出错: {e}")
                # 计算特征维度
                dim = self.bins * 3  # 颜色直方图
                if self.use_gradient:
                    dim += self.bins * 2  # 梯度直方图
                features.append(np.zeros(dim))

        return np.array(features)


class TextureFeatureExtractor(BaseFeatureExtractor):
    """
    纹理特征提取器
    提取LBP、GLCM等纹理描述符
    """

    def __init__(self, method='lbp', resize_dim=(256, 256)):
        """
        Parameters
        ----------
        method : str
            纹理特征方法 ('lbp', 'glcm', 'gabor')
        resize_dim : tuple
            图像调整大小
        """
        self.method = method
        self.resize_dim = resize_dim

    def _compute_lbp(self, gray_img):
        """计算局部二值模式(LBP)特征"""
        radius = 1
        n_points = 8 * radius

        h, w = gray_img.shape
        lbp = np.zeros_like(gray_img)

        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray_img[i, j]
                code = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = i + radius * np.cos(angle)
                    y = j + radius * np.sin(angle)

                    # 双线性插值
                    x1, y1 = int(x), int(y)
                    x2, y2 = min(x1 + 1, h - 1), min(y1 + 1, w - 1)

                    fx, fy = x - x1, y - y1
                    val = (1 - fx) * (1 - fy) * gray_img[x1, y1] + \
                          fx * (1 - fy) * gray_img[x2, y1] + \
                          (1 - fx) * fy * gray_img[x1, y2] + \
                          fx * fy * gray_img[x2, y2]

                    if val >= center:
                        code |= (1 << k)

                lbp[i, j] = code

        # 计算LBP直方图
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-6)

        return hist

    def extract(self, image_paths: Union[List[str], np.ndarray]) -> np.ndarray:
        features = []

        for path in tqdm(image_paths, desc="提取纹理特征"):
            try:
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"无法读取图像: {path}")

                img = cv2.resize(img, self.resize_dim)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if self.method == 'lbp':
                    feature = self._compute_lbp(gray)
                else:
                    # 简单的纹理统计特征
                    feature = []
                    # 全局统计
                    feature.extend([
                        gray.mean() / 255.0,
                        gray.std() / 255.0,
                        np.percentile(gray, 25) / 255.0,
                        np.percentile(gray, 50) / 255.0,
                        np.percentile(gray, 75) / 255.0
                    ])

                    # 局部统计（分块）
                    h, w = gray.shape
                    block_h, block_w = h // 4, w // 4
                    for i in range(4):
                        for j in range(4):
                            block = gray[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
                            feature.extend([
                                block.mean() / 255.0,
                                block.std() / 255.0
                            ])

                    feature = np.array(feature)

                features.append(feature)

            except Exception as e:
                print(f"处理 {path} 出错: {e}")
                if self.method == 'lbp':
                    features.append(np.zeros(256))
                else:
                    features.append(np.zeros(5 + 4 * 4 * 2))

        return np.array(features)


class EdgeFeatureExtractor(BaseFeatureExtractor):
    """
    边缘特征提取器
    提取边缘密度、轮廓特征等
    """

    def __init__(self, resize_dim=(256, 256), use_canny=True, use_contour=True):
        self.resize_dim = resize_dim
        self.use_canny = use_canny
        self.use_contour = use_contour

    def extract(self, image_paths: Union[List[str], np.ndarray]) -> np.ndarray:
        features = []

        for path in tqdm(image_paths, desc="提取边缘特征"):
            try:
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"无法读取图像: {path}")

                img = cv2.resize(img, self.resize_dim)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                feature_vec = []

                if self.use_canny:
                    # Canny边缘检测
                    edges = cv2.Canny(gray, 50, 150)

                    # 全局边缘密度
                    edge_density = np.sum(edges > 0) / edges.size
                    feature_vec.append(edge_density)

                    # 分块边缘密度
                    h, w = edges.shape
                    block_h, block_w = h // 3, w // 3
                    for i in range(3):
                        for j in range(3):
                            block = edges[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
                            block_density = np.sum(block > 0) / block.size
                            feature_vec.append(block_density)

                if self.use_contour:
                    # 轮廓特征
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # 轮廓数量
                    feature_vec.append(len(contours) / 100.0)  # 归一化

                    # 最大轮廓面积比例
                    if contours:
                        areas = [cv2.contourArea(c) for c in contours]
                        max_area_ratio = max(areas) / (img.shape[0] * img.shape[1])
                        feature_vec.append(max_area_ratio)
                    else:
                        feature_vec.append(0)

                features.append(feature_vec)

            except Exception as e:
                print(f"处理 {path} 出错: {e}")
                dim = 0
                if self.use_canny:
                    dim += 1 + 9  # 全局 + 3x3块
                if self.use_contour:
                    dim += 2
                features.append(np.zeros(dim))

        return np.array(features)


class CombinedFeatureExtractor(BaseFeatureExtractor):
    """
    组合特征提取器
    可以组合多个特征提取器
    """

    def __init__(self, extractors: List[BaseFeatureExtractor]):
        """
        Parameters
        ----------
        extractors : List[BaseFeatureExtractor]
            要组合的特征提取器列表
        """
        self.extractors = extractors

    def extract(self, image_paths: Union[List[str], np.ndarray]) -> np.ndarray:
        all_features = []

        for extractor in self.extractors:
            features = extractor.extract(image_paths)
            all_features.append(features)

        # 水平拼接所有特征
        return np.hstack(all_features)


# ============================================================================
#                         通用VisionOCSVM检测器
# ============================================================================

class VisionOCSVM(BaseVisionDetector):
    """
    通用的基于One-Class SVM的视觉异常检测器
    支持插入不同的特征提取器，适用于各种视觉异常检测任务
    """

    def __init__(self,
                 feature_extractor: Optional[BaseFeatureExtractor] = None,
                 contamination: float = 0.1,
                 nu: float = 0.1,
                 kernel: str = 'rbf',
                 gamma: Union[str, float] = 'auto',
                 use_scaler: bool = True,
                 use_pca: bool = False,
                 pca_components: Union[int, float] = 0.95,
                 random_state: int = 42):
        """
        Parameters
        ----------
        feature_extractor : BaseFeatureExtractor
            特征提取器实例
        contamination : float
            预期的异常比例
        nu : float
            One-Class SVM的nu参数
        kernel : str
            核函数类型
        gamma : str or float
            核函数系数
        use_scaler : bool
            是否使用标准化
        use_pca : bool
            是否使用PCA降维
        pca_components : int or float
            PCA组件数量或保留方差比例
        random_state : int
            随机种子
        """
        # 如果没有提供特征提取器，使用默认的
        if feature_extractor is None:
            feature_extractor = HistogramFeatureExtractor()
            print("未指定特征提取器，使用默认的HistogramFeatureExtractor")

        # 存储参数
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.use_scaler = use_scaler
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.random_state = random_state

        # 预处理组件
        self.scaler = StandardScaler() if use_scaler else None
        self.pca = None

        # 调用父类构造函数
        super(VisionOCSVM, self).__init__(
            contamination=contamination,
            feature_extractor=feature_extractor
        )

    def _build_detector(self):
        """构建One-Class SVM检测器"""
        return OCSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
            contamination=self.contamination
        )

    def fit(self, X: Union[List[str], np.ndarray], y=None):
        """
        训练检测器

        Parameters
        ----------
        X : List[str] or np.ndarray
            训练图像路径列表或特征矩阵
        y : None
            忽略（仅为API一致性）
        """
        print("=" * 60)
        print("开始训练 VisionOCSVM 检测器")
        print(f"参数配置:")
        print(f"  - Nu: {self.nu}")
        print(f"  - Kernel: {self.kernel}")
        print(f"  - Contamination: {self.contamination}")
        print(f"  - 使用标准化: {self.use_scaler}")
        print(f"  - 使用PCA: {self.use_pca}")
        print("=" * 60)

        # 提取特征
        if isinstance(X, list) and all(isinstance(x, str) for x in X):
            print(f"输入样本数: {len(X)}")
            features = self.feature_extractor.extract(X)
        else:
            features = X

        print(f"特征矩阵形状: {features.shape}")

        # 标准化
        if self.use_scaler:
            features = self.scaler.fit_transform(features)
            print("完成特征标准化")

        # PCA降维
        if self.use_pca:
            if isinstance(self.pca_components, float):
                # 保留指定比例的方差
                self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            else:
                # 使用指定数量的组件
                n_components = min(self.pca_components, features.shape[0] - 1, features.shape[1])
                self.pca = PCA(n_components=n_components, random_state=self.random_state)

            features = self.pca.fit_transform(features)
            variance_ratio = self.pca.explained_variance_ratio_.sum()
            print(f"PCA降维: {features.shape[1]} 维, 保留方差: {variance_ratio:.2%}")

        # 训练OCSVM
        print("训练 One-Class SVM...")
        self.detector.fit(features)

        # 同步训练分数
        self.decision_scores_ = self.detector.decision_scores_
        self._process_decision_scores()

        print("训练完成！")
        print("=" * 60)

        return self

    def decision_function(self, X: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        计算异常分数

        Parameters
        ----------
        X : List[str] or np.ndarray
            测试图像路径列表或特征矩阵

        Returns
        -------
        scores : np.ndarray
            异常分数（越高越异常）
        """
        # 提取特征
        if isinstance(X, list) and all(isinstance(x, str) for x in X):
            features = self.feature_extractor.extract(X)
        else:
            features = X

        # 确保是2D数组
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # 应用预处理
        if self.use_scaler and self.scaler is not None:
            features = self.scaler.transform(features)

        if self.use_pca and self.pca is not None:
            features = self.pca.transform(features)

        # 计算异常分数
        return self.detector.decision_function(features)

    def save(self, filepath: str):
        """保存模型"""
        model_data = {
            'detector': self.detector,
            'feature_extractor': self.feature_extractor,
            'scaler': self.scaler,
            'pca': self.pca,
            'threshold_': self.threshold_,
            'contamination': self.contamination,
            'nu': self.nu,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'use_scaler': self.use_scaler,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components
        }
        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)

        # 创建实例
        detector = cls(
            feature_extractor=model_data['feature_extractor'],
            contamination=model_data['contamination'],
            nu=model_data['nu'],
            kernel=model_data['kernel'],
            gamma=model_data['gamma'],
            use_scaler=model_data['use_scaler'],
            use_pca=model_data['use_pca'],
            pca_components=model_data['pca_components']
        )

        # 恢复训练好的组件
        detector.detector = model_data['detector']
        detector.scaler = model_data['scaler']
        detector.pca = model_data['pca']
        detector.threshold_ = model_data['threshold_']

        return detector


# ============================================================================
#                          便捷工厂函数
# ============================================================================

def create_detector(task: str = 'default', **kwargs) -> VisionOCSVM:
    """
    根据任务类型创建预配置的检测器

    Parameters
    ----------
    task : str
        任务类型 ('default', 'texture', 'color', 'edge', 'structure', 'combined')
    **kwargs
        传递给VisionOCSVM的额外参数

    Returns
    -------
    detector : VisionOCSVM
        配置好的检测器
    """
    # 预定义的特征提取器配置
    extractors = {
        'default': HistogramFeatureExtractor(),
        'texture': TextureFeatureExtractor(),
        'color': HistogramFeatureExtractor(color_space='HSV'),
        'edge': EdgeFeatureExtractor(),
        'pixel': PixelFeatureExtractor(),
        'combined': CombinedFeatureExtractor([
            HistogramFeatureExtractor(bins=16),
            TextureFeatureExtractor(),
            EdgeFeatureExtractor()
        ])
    }

    # 获取对应的特征提取器
    if task in extractors:
        feature_extractor = extractors[task]
    else:
        print(f"未知任务类型 '{task}'，使用默认配置")
        feature_extractor = extractors['default']

    # 创建检测器
    return VisionOCSVM(feature_extractor=feature_extractor, **kwargs)


# ============================================================================
#                            使用示例
# ============================================================================

if __name__ == "__main__":
    # 示例1: 使用预定义配置
    print("示例1: 纹理异常检测")
    detector1 = create_detector('texture', nu=0.1, contamination=0.05)
    # detector1.fit(normal_image_paths)
    # predictions = detector1.predict(test_image_paths)

    # 示例2: 自定义特征提取器
    print("\n示例2: 自定义组合特征")
    custom_extractor = CombinedFeatureExtractor([
        HistogramFeatureExtractor(bins=64, color_space='HSV'),
        EdgeFeatureExtractor(use_contour=True),
        TextureFeatureExtractor(method='lbp')
    ])
    detector2 = VisionOCSVM(
        feature_extractor=custom_extractor,
        nu=0.05,
        use_pca=True,
        pca_components=50
    )
    # detector2.fit(normal_image_paths)

    # 示例3: 简单像素级检测
    print("\n示例3: 像素级异常检测")
    detector3 = VisionOCSVM(
        feature_extractor=PixelFeatureExtractor(resize_dim=(32, 32)),
        nu=0.1,
        use_pca=True,
        pca_components=0.95  # 保留95%的方差
    )
    # detector3.fit(normal_image_paths)

    print("\n框架初始化成功！")
    print("可用的预定义任务: 'texture', 'color', 'edge', 'pixel', 'combined'")