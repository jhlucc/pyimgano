# -*- coding: utf-8 -*-
"""
Vision CBLOF - 基于聚类的视觉异常检测器
遵循 BaseVisionDetector 架构，不依赖PyOD的基础类
"""

import warnings
import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array, check_is_fitted
from tqdm import tqdm
import matplotlib.pyplot as plt

# 只从本地基类导入
from .baseml import BaseVisionDetector
from .registry import register_model


# ===================================================================
#                     工具函数（替代PyOD的工具）
# ===================================================================

def check_parameter(param, low=None, high=None, param_name='parameter',
                    include_left=True, include_right=True):
    """参数范围检查"""
    if low is not None:
        if include_left:
            if param < low:
                raise ValueError(f"{param_name} = {param} 必须 >= {low}")
        else:
            if param <= low:
                raise ValueError(f"{param_name} = {param} 必须 > {low}")

    if high is not None:
        if include_right:
            if param > high:
                raise ValueError(f"{param_name} = {param} 必须 <= {high}")
        else:
            if param >= high:
                raise ValueError(f"{param_name} = {param} 必须 < {high}")


def process_decision_scores(scores, contamination):
    """
    处理决策分数，计算阈值和标签

    Parameters
    ----------
    scores : array-like
        异常分数
    contamination : float
        污染率

    Returns
    -------
    threshold : float
        决策阈值
    labels : array
        二分类标签（0=正常，1=异常）
    """
    scores = np.asarray(scores)
    n_samples = len(scores)
    n_outliers = int(n_samples * contamination)

    # 计算阈值
    threshold = np.percentile(scores, 100 * (1 - contamination))

    # 生成标签
    labels = (scores > threshold).astype(int)

    # 确保恰好有n_outliers个异常
    if labels.sum() != n_outliers:
        # 根据分数排序，标记top-k个为异常
        sorted_indices = np.argsort(scores)
        labels = np.zeros(n_samples, dtype=int)
        labels[sorted_indices[-n_outliers:]] = 1
        threshold = scores[sorted_indices[-n_outliers]]

    return threshold, labels


# ===================================================================
#                     特征提取器类
# ===================================================================

class ImageFeatureExtractor:
    """
    图像特征提取器 - 实现extract方法供BaseVisionDetector使用

    Parameters
    ----------
    method : str, optional (default='combined')
        特征提取方法：'color', 'texture', 'deep', 'combined'

    reduce_dim : bool, optional (default=True)
        是否进行PCA降维

    n_components : int, optional (default=50)
        PCA降维后的维度
    """

    def __init__(self, method='combined', reduce_dim=True, n_components=50):
        self.method = method
        self.reduce_dim = reduce_dim
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components) if reduce_dim else None
        self.is_fitted = False

    def extract(self, X):
        """
        提取图像特征 - BaseVisionDetector要求的接口

        Parameters
        ----------
        X : list of str or numpy array
            图像文件路径列表或特征数组

        Returns
        -------
        features : numpy array
            提取的特征矩阵
        """
        # 如果已经是特征矩阵，直接处理
        if isinstance(X, np.ndarray):
            if self.is_fitted:
                X = self.scaler.transform(X)
                if self.reduce_dim and self.pca is not None:
                    X = self.pca.transform(X)
            return X

        # 从图像路径提取特征
        print("提取图像特征...")
        features = []

        for img_path in tqdm(X):
            try:
                feat = self._extract_single_image_features(img_path)
                features.append(feat)
            except Exception as e:
                print(f"处理 {img_path} 出错: {e}")
                # 添加零特征
                if len(features) > 0:
                    features.append(np.zeros_like(features[0]))
                else:
                    features.append(np.zeros(100))

        features = np.array(features)

        # 标准化和降维
        if not self.is_fitted:
            features = self.scaler.fit_transform(features)
            if self.reduce_dim and features.shape[1] > self.n_components:
                features = self.pca.fit_transform(features)
                print(f"PCA降维: {features.shape[1]} 维")
                print(f"保留方差: {self.pca.explained_variance_ratio_.sum():.2%}")
            self.is_fitted = True
        else:
            features = self.scaler.transform(features)
            if self.reduce_dim and self.pca is not None:
                features = self.pca.transform(features)

        return features

    def _extract_single_image_features(self, image_path):
        """提取单张图像的特征"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        features = []

        if self.method in ['color', 'combined']:
            features.extend(self._extract_color_features(img))

        if self.method in ['texture', 'combined']:
            features.extend(self._extract_texture_features(img))

        if self.method == 'deep':
            features.extend(self._extract_deep_features(img))

        return np.array(features)

    def _extract_color_features(self, img):
        """提取颜色特征"""
        features = []

        # RGB直方图
        for i in range(3):
            hist, _ = np.histogram(img[:, :, i], bins=16, range=(0, 256))
            hist = hist.astype(float) / (hist.sum() + 1e-6)
            features.extend(hist)

        # 颜色统计
        for i in range(3):
            channel = img[:, :, i]
            features.extend([
                channel.mean() / 255.0,
                channel.std() / 255.0,
                np.percentile(channel, 25) / 255.0,
                np.percentile(channel, 75) / 255.0
            ])

        # HSV特征
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i in range(3):
            features.extend([
                hsv[:, :, i].mean() / 255.0,
                hsv[:, :, i].std() / 255.0
            ])

        return features

    def _extract_texture_features(self, img):
        """提取纹理特征"""
        features = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 梯度特征
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        features.extend([
            grad_mag.mean() / 255.0,
            grad_mag.std() / 255.0,
            (grad_mag > 50).sum() / grad_mag.size
        ])

        # 简单纹理统计
        features.extend([
            gray.mean() / 255.0,
            gray.std() / 255.0
        ])

        # 频域特征
        fft = np.fft.fft2(cv2.resize(gray, (64, 64)))
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        center = 32
        low_freq = magnitude[center - 8:center + 8, center - 8:center + 8].sum()
        total_freq = magnitude.sum()
        features.append(low_freq / (total_freq + 1e-6))

        return features

    def _extract_deep_features(self, img):
        """提取深度特征（简化版）"""
        img_small = cv2.resize(img, (16, 16))
        features = img_small.flatten() / 255.0
        return features[:100]


# ===================================================================
#                     核心CBLOF算法（独立实现）
# ===================================================================

class CoreCBLOF:
    """
    核心CBLOF算法 - 独立实现，不依赖PyOD

    Parameters
    ----------
    n_clusters : int
        聚类数量
    contamination : float
        污染率
    alpha : float
        大簇样本占比阈值
    beta : float
        簇大小比例阈值
    use_weights : bool
        是否使用簇大小作为权重
    random_state : int
        随机种子
    """

    def __init__(self,
                 n_clusters=8,
                 contamination=0.1,
                 alpha=0.9,
                 beta=5,
                 use_weights=False,
                 random_state=None):

        self.n_clusters = n_clusters
        self.contamination = contamination
        self.alpha = alpha
        self.beta = beta
        self.use_weights = use_weights
        self.random_state = random_state

        # 将在fit时设置
        self.clustering_estimator_ = None
        self.cluster_labels_ = None
        self.cluster_centers_ = None
        self.cluster_sizes_ = None
        self.n_clusters_ = None
        self.large_cluster_labels_ = None
        self.small_cluster_labels_ = None
        self.decision_scores_ = None
        self.threshold_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        """训练CBLOF模型"""
        X = check_array(X)
        n_samples, n_features = X.shape

        # 参数验证
        check_parameter(self.alpha, low=0, high=1, param_name='alpha',
                        include_left=False, include_right=False)
        check_parameter(self.beta, low=1, param_name='beta',
                        include_left=False)

        # 执行聚类
        self.clustering_estimator_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )
        self.clustering_estimator_.fit(X)

        # 获取聚类结果
        self.cluster_labels_ = self.clustering_estimator_.labels_
        self.cluster_centers_ = self.clustering_estimator_.cluster_centers_
        self.cluster_sizes_ = np.bincount(self.cluster_labels_)
        self.n_clusters_ = len(self.cluster_sizes_)

        if self.n_clusters_ != self.n_clusters:
            warnings.warn(f"实际聚类数 {self.n_clusters_} 与设定值 {self.n_clusters} 不同")

        # 区分大小簇
        self._set_small_large_clusters(n_samples)

        # 计算异常分数
        self.decision_scores_ = self._compute_scores(X, self.cluster_labels_)

        # 计算阈值和标签
        self.threshold_, self.labels_ = process_decision_scores(
            self.decision_scores_, self.contamination
        )

        return self

    def decision_function(self, X):
        """计算异常分数"""
        check_is_fitted(self, ['cluster_centers_', 'threshold_'])
        X = check_array(X)

        # 预测聚类标签
        labels = self.clustering_estimator_.predict(X)

        # 计算异常分数
        return self._compute_scores(X, labels)

    def predict(self, X):
        """预测标签"""
        scores = self.decision_function(X)
        return (scores > self.threshold_).astype(int)

    def _set_small_large_clusters(self, n_samples):
        """区分大簇和小簇"""
        # 按簇大小排序（从大到小）
        sorted_indices = np.argsort(self.cluster_sizes_)[::-1]

        alpha_list = []
        beta_list = []

        for i in range(1, self.n_clusters_):
            # α条件：前i个簇的样本数占比
            temp_sum = np.sum(self.cluster_sizes_[sorted_indices[:i]])
            if temp_sum >= n_samples * self.alpha:
                alpha_list.append(i)

            # β条件：相邻簇大小比例
            ratio = self.cluster_sizes_[sorted_indices[i - 1]] / (self.cluster_sizes_[sorted_indices[i]] + 1e-10)
            if ratio >= self.beta:
                beta_list.append(i)

        # 找到同时满足条件的分割点
        intersection = np.intersect1d(alpha_list, beta_list)

        if len(intersection) > 0:
            threshold = intersection[0]
        elif len(alpha_list) > 0:
            threshold = alpha_list[0]
        elif len(beta_list) > 0:
            threshold = beta_list[0]
        else:
            threshold = 1
            warnings.warn("无法有效区分大小簇，使用默认设置")

        self.large_cluster_labels_ = sorted_indices[:threshold]
        self.small_cluster_labels_ = sorted_indices[threshold:]

        print(f"聚类分析:")
        print(f"  大簇数量: {len(self.large_cluster_labels_)}")
        print(f"  小簇数量: {len(self.small_cluster_labels_)}")

    def _compute_scores(self, X, labels):
        """计算异常分数"""
        scores = np.zeros(X.shape[0])

        # 小簇中的样本：计算到最近大簇中心的距离
        small_mask = np.isin(labels, self.small_cluster_labels_)
        if small_mask.any():
            large_centers = self.cluster_centers_[self.large_cluster_labels_]
            dist_to_large = cdist(X[small_mask], large_centers)
            scores[small_mask] = np.min(dist_to_large, axis=1)

        # 大簇中的样本：计算到所属簇中心的距离
        large_mask = np.isin(labels, self.large_cluster_labels_)
        if large_mask.any():
            for label in self.large_cluster_labels_:
                label_mask = (labels == label)
                if label_mask.any():
                    center = self.cluster_centers_[label]
                    scores[label_mask] = np.linalg.norm(X[label_mask] - center, axis=1)

        # 使用簇大小作为权重
        if self.use_weights:
            scores = scores * self.cluster_sizes_[labels]

        return scores


# ===================================================================
#                     VisionCBLOF（主类）
# ===================================================================

@register_model(
    "vision_cblof",
    tags=("vision", "classical", "clustering"),
    metadata={"description": "基于 CBLOF 的视觉异常检测器"},
)
class VisionCBLOF(BaseVisionDetector):
    """
    基于CBLOF算法的视觉异常检测器

    继承自BaseVisionDetector，符合统一的接口规范

    Parameters
    ----------
    contamination : float, optional (default=0.1)
        数据集中异常样本的比例

    feature_extractor : object, optional
        特征提取器实例，必须有extract方法

    n_clusters : int, optional (default=8)
        聚类数量

    alpha : float, optional (default=0.9)
        区分大小簇的系数（大簇样本占比）

    beta : float, optional (default=5)
        区分大小簇的系数（簇大小比例）

    use_weights : bool, optional (default=False)
        是否使用簇大小作为权重

    feature_method : str, optional (default='combined')
        特征提取方法（仅在未提供feature_extractor时使用）

    reduce_dim : bool, optional (default=True)
        是否PCA降维（仅在未提供feature_extractor时使用）

    n_components : int, optional (default=50)
        PCA维度（仅在未提供feature_extractor时使用）

    random_state : int, optional
        随机种子

    Examples
    --------
    >>> from vision_cblof import VisionCBLOF
    >>> # 使用默认特征提取器
    >>> detector = VisionCBLOF(n_clusters=8, contamination=0.1)
    >>> detector.fit(train_image_paths)
    >>> scores = detector.decision_function(test_image_paths)
    >>> labels = detector.predict(test_image_paths)
    """

    def __init__(self,
                 contamination=0.1,
                 feature_extractor=None,
                 n_clusters=8,
                 alpha=0.9,
                 beta=5,
                 use_weights=False,
                 feature_method='combined',
                 reduce_dim=True,
                 n_components=50,
                 random_state=None):

        # 保存CBLOF特定参数
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.use_weights = use_weights
        self.random_state = random_state

        # 如果未提供特征提取器，创建默认的
        if feature_extractor is None:
            feature_extractor = ImageFeatureExtractor(
                method=feature_method,
                reduce_dim=reduce_dim,
                n_components=n_components
            )
            print(f"使用默认特征提取器: method={feature_method}, PCA={reduce_dim}")

        # 调用父类构造函数
        super(VisionCBLOF, self).__init__(
            contamination=contamination,
            feature_extractor=feature_extractor
        )

    def _build_detector(self):
        """
        构建核心检测器实例
        BaseVisionDetector要求的接口
        """
        return CoreCBLOF(
            n_clusters=self.n_clusters,
            contamination=self.contamination,
            alpha=self.alpha,
            beta=self.beta,
            use_weights=self.use_weights,
            random_state=self.random_state
        )

    def visualize_clusters(self):
        """可视化聚类结果"""
        if not hasattr(self.detector, 'cluster_labels_'):
            print("请先训练模型")
            return

        # 获取聚类信息
        n_clusters = self.detector.n_clusters_
        cluster_sizes = self.detector.cluster_sizes_
        large_clusters = self.detector.large_cluster_labels_
        small_clusters = self.detector.small_cluster_labels_

        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 簇大小分布
        colors = ['green' if i in large_clusters else 'orange'
                  for i in range(n_clusters)]

        ax1.bar(range(n_clusters), cluster_sizes, color=colors)
        ax1.set_xlabel('簇索引')
        ax1.set_ylabel('簇大小')
        ax1.set_title('簇大小分布')
        ax1.legend(['大簇', '小簇'])

        # 异常分数分布
        if hasattr(self, 'decision_scores_'):
            ax2.hist(self.decision_scores_, bins=30, edgecolor='black')
            ax2.axvline(self.detector.threshold_, color='r',
                        linestyle='--', label=f'阈值={self.detector.threshold_:.3f}')
            ax2.set_xlabel('异常分数')
            ax2.set_ylabel('频数')
            ax2.set_title('异常分数分布')
            ax2.legend()

        plt.tight_layout()
        plt.show()

    def get_cluster_info(self):
        """获取聚类详细信息"""
        if not hasattr(self.detector, 'cluster_labels_'):
            return {"status": "模型未训练"}

        info = {
            "算法": "Vision-CBLOF",
            "聚类数": self.detector.n_clusters_,
            "大簇": list(self.detector.large_cluster_labels_),
            "小簇": list(self.detector.small_cluster_labels_),
            "簇大小": {i: int(self.detector.cluster_sizes_[i])
                       for i in range(self.detector.n_clusters_)},
            "α参数": self.alpha,
            "β参数": self.beta,
            "使用权重": self.use_weights,
            "污染率": self.contamination,
            "阈值": float(self.detector.threshold_)
        }

        # 添加异常检测结果统计
        if hasattr(self.detector, 'labels_'):
            n_anomalies = self.detector.labels_.sum()
            n_total = len(self.detector.labels_)
            info["训练集异常数"] = f"{n_anomalies}/{n_total} ({n_anomalies / n_total * 100:.1f}%)"

        return info


# ===================================================================
#                          使用示例
# ===================================================================

if __name__ == "__main__":
    print("Vision-CBLOF 异常检测器示例")
    print("=" * 60)

    # 示例1: 使用默认配置
    print("\n示例1: 默认配置")
    detector = VisionCBLOF(
        n_clusters=8,
        contamination=0.1,
        feature_method='combined'
    )
    print("创建成功，特征方法: combined")

    # 示例2: 自定义特征提取器
    print("\n示例2: 自定义特征提取器")
    custom_extractor = ImageFeatureExtractor(
        method='texture',
        reduce_dim=False
    )

    detector2 = VisionCBLOF(
        feature_extractor=custom_extractor,
        n_clusters=5,
        alpha=0.8,
        beta=3
    )
    print("创建成功，特征方法: texture, 不使用PCA")

    # 示例3: 使用模拟数据演示完整流程
    print("\n示例3: 模拟数据演示")
    print("-" * 40)


    # 创建模拟特征提取器
    class MockExtractor:
        def extract(self, X):
            # 模拟正常数据和异常数据
            np.random.seed(42)
            n_samples = len(X) if isinstance(X, list) else X.shape[0]
            n_features = 20

            # 90%正常数据（聚类分布）
            n_normal = int(n_samples * 0.9)
            normal_data = np.random.randn(n_normal, n_features)

            # 10%异常数据（离群点）
            n_anomaly = n_samples - n_normal
            anomaly_data = np.random.randn(n_anomaly, n_features) * 3 + 5

            data = np.vstack([normal_data, anomaly_data])
            return data


    # 创建检测器
    mock_detector = VisionCBLOF(
        feature_extractor=MockExtractor(),
        n_clusters=3,
        contamination=0.1
    )

    # 训练
    train_paths = [f"img_{i}.jpg" for i in range(100)]
    print("训练模型...")
    mock_detector.fit(train_paths)

    # 测试
    test_paths = [f"test_{i}.jpg" for i in range(20)]
    scores = mock_detector.decision_function(test_paths)
    labels = mock_detector.predict(test_paths)

    print(f"\n检测结果:")
    print(f"  异常分数范围: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  检测到异常: {labels.sum()}/{len(labels)}")

    # 显示聚类信息
    info = mock_detector.get_cluster_info()
    print(f"\n模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("示例完成")
