# -*- coding: utf-8 -*-
"""
Vision LODA - 基于LODA算法的视觉异常检测器
遵循 BaseVisionDetector 架构，不依赖PyOD基类
"""

import numbers
import warnings
import numpy as np
import cv2
import os
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt

# 只从本地基类导入
from .baseml import BaseVisionDetector


# ===================================================================
#                     工具函数
# ===================================================================

def get_optimal_n_bins(data, max_bins=50):
    """
    使用Birge-Rozenblac方法自动确定最优bin数量

    Parameters
    ----------
    data : array-like
        输入数据
    max_bins : int
        最大bin数量

    Returns
    -------
    optimal_bins : int
        最优bin数量
    """
    n = len(data)
    if n < 10:
        return min(5, n)

    # 简化版的Birge-Rozenblac方法
    # 使用Sturges规则作为起点
    sturges_bins = int(np.ceil(np.log2(n) + 1))

    # 使用Scott规则
    std_dev = np.std(data)
    if std_dev > 0:
        bin_width = 3.5 * std_dev / (n ** (1 / 3))
        data_range = np.max(data) - np.min(data)
        scott_bins = int(np.ceil(data_range / bin_width))
    else:
        scott_bins = sturges_bins

    # 取两者的平均值，并限制在合理范围内
    optimal_bins = int((sturges_bins + scott_bins) / 2)
    optimal_bins = max(5, min(optimal_bins, max_bins))

    return optimal_bins


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
        二分类标签
    """
    scores = np.asarray(scores)
    n_samples = len(scores)
    n_outliers = int(n_samples * contamination)

    # 计算阈值
    threshold = np.percentile(scores, 100 * (1 - contamination))

    # 生成标签
    labels = (scores > threshold).astype(int)

    # 确保异常数量正确
    if labels.sum() != n_outliers and n_outliers > 0:
        sorted_indices = np.argsort(scores)
        labels = np.zeros(n_samples, dtype=int)
        if n_outliers > 0:
            labels[sorted_indices[-n_outliers:]] = 1
            threshold = scores[sorted_indices[-n_outliers]]

    return threshold, labels


# ===================================================================
#                     特征提取器
# ===================================================================

class LODAFeatureExtractor:
    """
    LODA专用的图像特征提取器

    Parameters
    ----------
    method : str, optional (default='histogram')
        特征提取方法：'histogram', 'statistical', 'combined'

    normalize : bool, optional (default=True)
        是否标准化特征
    """

    def __init__(self, method='histogram', normalize=True):
        self.method = method
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.is_fitted = False

    def extract(self, X):
        """
        提取图像特征 - BaseVisionDetector要求的接口

        Parameters
        ----------
        X : list of str or numpy array
            图像路径列表或特征数组

        Returns
        -------
        features : numpy array
            提取的特征矩阵
        """
        # 如果已经是特征矩阵，直接处理
        if isinstance(X, np.ndarray):
            if self.normalize and self.is_fitted:
                X = self.scaler.transform(X)
            return X

        # 从图像路径提取特征
        print("提取LODA特征...")
        features = []

        for img_path in tqdm(X):
            try:
                feat = self._extract_single_image(img_path)
                features.append(feat)
            except Exception as e:
                print(f"处理 {img_path} 出错: {e}")
                # 添加零特征
                if len(features) > 0:
                    features.append(np.zeros_like(features[0]))
                else:
                    features.append(np.zeros(100))

        features = np.array(features)

        # 标准化
        if self.normalize:
            if not self.is_fitted:
                features = self.scaler.fit_transform(features)
                self.is_fitted = True
            else:
                features = self.scaler.transform(features)

        return features

    def _extract_single_image(self, image_path):
        """提取单张图像的特征"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        features = []

        if self.method in ['histogram', 'combined']:
            # 直方图特征
            for i in range(3):
                hist, _ = np.histogram(img[:, :, i], bins=32, range=(0, 256))
                hist = hist.astype(float) / (hist.sum() + 1e-6)
                features.extend(hist)

        if self.method in ['statistical', 'combined']:
            # 统计特征
            for i in range(3):
                channel = img[:, :, i].flatten()
                features.extend([
                    channel.mean() / 255.0,
                    channel.std() / 255.0,
                    np.percentile(channel, 25) / 255.0,
                    np.percentile(channel, 50) / 255.0,
                    np.percentile(channel, 75) / 255.0,
                    channel.min() / 255.0,
                    channel.max() / 255.0
                ])

            # 添加简单的纹理特征
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

            features.extend([
                grad_mag.mean() / 255.0,
                grad_mag.std() / 255.0
            ])

        return np.array(features)


# ===================================================================
#                     核心LODA算法
# ===================================================================

class CoreLODA:
    """
    核心LODA算法实现 - 轻量级在线异常检测

    LODA使用随机投影和直方图来检测异常

    Parameters
    ----------
    contamination : float
        污染率
    n_bins : int or str
        直方图的bin数量，如果是'auto'则自动确定
    n_random_cuts : int
        随机切割数量
    """

    def __init__(self, contamination=0.1, n_bins=10, n_random_cuts=100):
        self.contamination = contamination
        self.n_bins = n_bins
        self.n_random_cuts = n_random_cuts
        self.weights = np.ones(n_random_cuts, dtype=float) / n_random_cuts

        # 将在fit时设置
        self.projections_ = None
        self.histograms_ = None
        self.limits_ = None
        self.n_bins_ = None  # 用于auto模式
        self.decision_scores_ = None
        self.threshold_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        """训练LODA模型"""
        X = check_array(X)
        n_samples, n_components = X.shape

        pred_scores = np.zeros([n_samples, 1])

        # 生成稀疏随机投影
        n_nonzero_components = int(np.sqrt(n_components))
        n_zero_components = n_components - n_nonzero_components

        self.projections_ = np.random.randn(self.n_random_cuts, n_components)

        # 处理不同的n_bins设置
        if isinstance(self.n_bins, str) and self.n_bins.lower() == "auto":
            # 自动确定bin数量
            self.histograms_ = []
            self.limits_ = []
            self.n_bins_ = []

            print("使用自动bin数量确定...")
            for i in range(self.n_random_cuts):
                # 随机设置一些分量为0（稀疏投影）
                rands = np.random.permutation(n_components)[:n_zero_components]
                self.projections_[i, rands] = 0.

                # 投影数据
                projected_data = self.projections_[i, :].dot(X.T)

                # 确定最优bin数量
                n_bins = get_optimal_n_bins(projected_data)
                self.n_bins_.append(n_bins)

                # 创建直方图
                histogram, limits = np.histogram(
                    projected_data, bins=n_bins, density=False)
                histogram = histogram.astype(np.float64)
                histogram += 1e-12  # 避免log(0)
                histogram /= np.sum(histogram)

                self.histograms_.append(histogram)
                self.limits_.append(limits)

                # 计算训练样本的分数
                inds = np.searchsorted(limits[:n_bins - 1],
                                       projected_data, side='left')
                pred_scores[:, 0] += -self.weights[i] * np.log(histogram[inds])

        elif isinstance(self.n_bins, numbers.Integral):
            # 固定bin数量
            self.histograms_ = np.zeros((self.n_random_cuts, self.n_bins))
            self.limits_ = np.zeros((self.n_random_cuts, self.n_bins + 1))

            print(f"使用固定bin数量: {self.n_bins}")
            for i in range(self.n_random_cuts):
                # 随机设置一些分量为0
                rands = np.random.permutation(n_components)[:n_zero_components]
                self.projections_[i, rands] = 0.

                # 投影数据
                projected_data = self.projections_[i, :].dot(X.T)

                # 创建直方图
                self.histograms_[i, :], self.limits_[i, :] = np.histogram(
                    projected_data, bins=self.n_bins, density=False)
                self.histograms_[i, :] += 1e-12
                self.histograms_[i, :] /= np.sum(self.histograms_[i, :])

                # 计算训练样本的分数
                inds = np.searchsorted(self.limits_[i, :self.n_bins - 1],
                                       projected_data, side='left')
                pred_scores[:, 0] += -self.weights[i] * np.log(
                    self.histograms_[i, inds])
        else:
            raise ValueError(f"n_bins必须是整数或'auto'，得到: {self.n_bins}")

        # 计算最终分数
        self.decision_scores_ = (pred_scores / self.n_random_cuts).ravel()

        # 计算阈值和标签
        self.threshold_, self.labels_ = process_decision_scores(
            self.decision_scores_, self.contamination
        )

        return self

    def decision_function(self, X):
        """计算异常分数"""
        check_is_fitted(self, ['projections_', 'decision_scores_'])

        X = check_array(X)
        pred_scores = np.zeros([X.shape[0], 1])

        if isinstance(self.n_bins, str) and self.n_bins.lower() == "auto":
            # 自动bin模式
            for i in range(self.n_random_cuts):
                projected_data = self.projections_[i, :].dot(X.T)

                inds = np.searchsorted(self.limits_[i][:self.n_bins_[i] - 1],
                                       projected_data, side='left')
                pred_scores[:, 0] += -self.weights[i] * np.log(
                    self.histograms_[i][inds])

        elif isinstance(self.n_bins, numbers.Integral):
            # 固定bin模式
            for i in range(self.n_random_cuts):
                projected_data = self.projections_[i, :].dot(X.T)

                inds = np.searchsorted(self.limits_[i, :self.n_bins - 1],
                                       projected_data, side='left')
                pred_scores[:, 0] += -self.weights[i] * np.log(
                    self.histograms_[i, inds])

        pred_scores /= self.n_random_cuts
        return pred_scores.ravel()

    def predict(self, X):
        """预测标签"""
        scores = self.decision_function(X)
        return (scores > self.threshold_).astype(int)


# ===================================================================
#                     VisionLODA主类
# ===================================================================

class VisionLODA(BaseVisionDetector):
    """
    基于LODA算法的视觉异常检测器

    LODA (Lightweight On-line Detector of Anomalies) 是一种轻量级的
    异常检测算法，使用随机投影和直方图来快速检测异常。

    Parameters
    ----------
    contamination : float, optional (default=0.1)
        数据集中异常样本的比例

    feature_extractor : object, optional
        特征提取器实例，必须有extract方法

    n_bins : int or str, optional (default=10)
        直方图的bin数量。如果设为'auto'，将自动确定最优数量

    n_random_cuts : int, optional (default=100)
        随机切割的数量

    feature_method : str, optional (default='histogram')
        特征提取方法（仅在未提供feature_extractor时使用）

    normalize_features : bool, optional (default=True)
        是否标准化特征（仅在未提供feature_extractor时使用）

    Examples
    --------
    >>> from vision_loda import VisionLODA
    >>> # 使用默认特征提取器
    >>> detector = VisionLODA(n_bins='auto', n_random_cuts=100)
    >>> detector.fit(train_image_paths)
    >>> scores = detector.decision_function(test_image_paths)
    >>> labels = detector.predict(test_image_paths)
    """

    def __init__(self,
                 contamination=0.1,
                 feature_extractor=None,
                 n_bins=10,
                 n_random_cuts=100,
                 feature_method='histogram',
                 normalize_features=True):

        # 保存LODA特定参数
        self.n_bins = n_bins
        self.n_random_cuts = n_random_cuts

        # 如果未提供特征提取器，创建默认的
        if feature_extractor is None:
            feature_extractor = LODAFeatureExtractor(
                method=feature_method,
                normalize=normalize_features
            )
            print(f"使用默认特征提取器: method={feature_method}, normalize={normalize_features}")

        # 调用父类构造函数
        super(VisionLODA, self).__init__(
            contamination=contamination,
            feature_extractor=feature_extractor
        )

    def _build_detector(self):
        """
        构建核心检测器实例
        BaseVisionDetector要求的接口
        """
        return CoreLODA(
            contamination=self.contamination,
            n_bins=self.n_bins,
            n_random_cuts=self.n_random_cuts
        )

    def visualize_projections(self, n_projections=5):
        """
        可视化随机投影的效果

        Parameters
        ----------
        n_projections : int
            要可视化的投影数量
        """
        if not hasattr(self.detector, 'projections_'):
            print("请先训练模型")
            return

        n_show = min(n_projections, self.n_random_cuts)
        fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))

        if n_show == 1:
            axes = [axes]

        for i in range(n_show):
            projection = self.detector.projections_[i]

            # 显示投影向量的强度
            axes[i].bar(range(len(projection)), np.abs(projection))
            axes[i].set_title(f'投影 {i + 1}')
            axes[i].set_xlabel('特征索引')
            axes[i].set_ylabel('投影权重绝对值')

        plt.tight_layout()
        plt.show()

    def visualize_histograms(self, n_histograms=5):
        """
        可视化直方图

        Parameters
        ----------
        n_histograms : int
            要可视化的直方图数量
        """
        if not hasattr(self.detector, 'histograms_'):
            print("请先训练模型")
            return

        n_show = min(n_histograms, self.n_random_cuts)
        fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))

        if n_show == 1:
            axes = [axes]

        for i in range(n_show):
            if isinstance(self.n_bins, str) and self.n_bins.lower() == "auto":
                # 自动bin模式
                histogram = self.detector.histograms_[i]
                n_bins = len(histogram)
            else:
                # 固定bin模式
                histogram = self.detector.histograms_[i]
                n_bins = self.n_bins

            axes[i].bar(range(n_bins), histogram)
            axes[i].set_title(f'直方图 {i + 1} (bins={n_bins})')
            axes[i].set_xlabel('Bin索引')
            axes[i].set_ylabel('概率')

        plt.tight_layout()
        plt.show()

    def visualize_scores(self):
        """可视化异常分数分布"""
        if not hasattr(self.detector, 'decision_scores_'):
            print("请先训练模型")
            return

        plt.figure(figsize=(10, 5))

        # 分数分布直方图
        plt.subplot(1, 2, 1)
        plt.hist(self.detector.decision_scores_, bins=30, edgecolor='black')
        plt.axvline(self.detector.threshold_, color='r',
                    linestyle='--', label=f'阈值={self.detector.threshold_:.3f}')
        plt.xlabel('异常分数')
        plt.ylabel('频数')
        plt.title('异常分数分布')
        plt.legend()

        # 分数排序图
        plt.subplot(1, 2, 2)
        sorted_scores = np.sort(self.detector.decision_scores_)
        n_samples = len(sorted_scores)
        colors = ['blue' if s <= self.detector.threshold_ else 'red'
                  for s in sorted_scores]

        plt.scatter(range(n_samples), sorted_scores, c=colors, s=10, alpha=0.5)
        plt.axhline(self.detector.threshold_, color='r',
                    linestyle='--', label=f'阈值={self.detector.threshold_:.3f}')
        plt.xlabel('样本索引（按分数排序）')
        plt.ylabel('异常分数')
        plt.title('排序后的异常分数')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def get_info(self):
        """获取模型信息"""
        if not hasattr(self.detector, 'projections_'):
            return {"status": "模型未训练"}

        info = {
            "算法": "Vision-LODA",
            "bin策略": self.n_bins,
            "随机切割数": self.n_random_cuts,
            "特征维度": self.detector.projections_.shape[1],
            "污染率": self.contamination,
            "阈值": float(self.detector.threshold_)
        }

        # 添加bin数量信息
        if isinstance(self.n_bins, str) and self.n_bins.lower() == "auto":
            info["bin数量范围"] = f"{min(self.detector.n_bins_)}-{max(self.detector.n_bins_)}"
            info["平均bin数"] = f"{np.mean(self.detector.n_bins_):.1f}"
        else:
            info["bin数量"] = self.n_bins

        # 添加检测结果统计
        if hasattr(self.detector, 'labels_'):
            n_anomalies = self.detector.labels_.sum()
            n_total = len(self.detector.labels_)
            info["训练集异常数"] = f"{n_anomalies}/{n_total} ({n_anomalies / n_total * 100:.1f}%)"

        return info


# ===================================================================
#                          使用示例
# ===================================================================

if __name__ == "__main__":
    print("Vision-LODA 异常检测器示例")
    print("=" * 60)

    # 示例1: 使用固定bin数量
    print("\n示例1: 固定bin数量")
    detector1 = VisionLODA(
        n_bins=10,
        n_random_cuts=100,
        contamination=0.1,
        feature_method='histogram'
    )
    print("创建成功，使用固定10个bins")

    # 示例2: 使用自动bin数量
    print("\n示例2: 自动确定bin数量")
    detector2 = VisionLODA(
        n_bins='auto',
        n_random_cuts=50,
        feature_method='combined'
    )
    print("创建成功，自动确定bin数量")

    # 示例3: 自定义特征提取器
    print("\n示例3: 自定义特征提取器")
    custom_extractor = LODAFeatureExtractor(
        method='statistical',
        normalize=False
    )
    detector3 = VisionLODA(
        feature_extractor=custom_extractor,
        n_bins=15,
        n_random_cuts=200
    )
    print("创建成功，使用统计特征，不标准化")

    # 示例4: 使用模拟数据演示完整流程
    print("\n示例4: 模拟数据演示")
    print("-" * 40)


    # 创建模拟特征提取器
    class MockExtractor:
        def extract(self, X):
            # 生成模拟数据
            np.random.seed(42)
            n_samples = len(X) if isinstance(X, list) else X.shape[0]
            n_features = 50

            # 90%正常数据（高斯分布）
            n_normal = int(n_samples * 0.9)
            normal_data = np.random.randn(n_normal, n_features)

            # 10%异常数据（均匀分布）
            n_anomaly = n_samples - n_normal
            anomaly_data = np.random.uniform(-5, 5, (n_anomaly, n_features))

            data = np.vstack([normal_data, anomaly_data])
            np.random.shuffle(data)
            return data


    # 创建检测器
    mock_detector = VisionLODA(
        feature_extractor=MockExtractor(),
        n_bins='auto',
        n_random_cuts=100,
        contamination=0.1
    )

    # 训练
    train_paths = [f"img_{i}.jpg" for i in range(500)]
    print("训练LODA模型...")
    mock_detector.fit(train_paths)

    # 测试
    test_paths = [f"test_{i}.jpg" for i in range(100)]
    scores = mock_detector.decision_function(test_paths)
    labels = mock_detector.predict(test_paths)

    print(f"\n检测结果:")
    print(f"  异常分数范围: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  平均分数: {scores.mean():.3f}")
    print(f"  检测到异常: {labels.sum()}/{len(labels)} ({labels.sum() / len(labels) * 100:.1f}%)")

    # 显示模型信息
    info = mock_detector.get_info()
    print(f"\n模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 可视化
    print("\n生成可视化...")
    # mock_detector.visualize_scores()  # 实际使用时取消注释
    # mock_detector.visualize_projections(3)  # 实际使用时取消注释
    # mock_detector.visualize_histograms(3)  # 实际使用时取消注释

    print("\n" + "=" * 60)
    print("示例完成")

    # 示例5: 性能对比
    print("\n示例5: 不同配置的性能对比")
    print("-" * 40)

    configs = [
        {"n_bins": 5, "name": "少量bins(5)"},
        {"n_bins": 20, "name": "中等bins(20)"},
        {"n_bins": 50, "name": "大量bins(50)"},
        {"n_bins": "auto", "name": "自动bins"}
    ]

    for config in configs:
        detector = VisionLODA(
            feature_extractor=MockExtractor(),
            n_bins=config["n_bins"],
            n_random_cuts=50,
            contamination=0.1
        )

        # 简单训练
        small_train = [f"img_{i}.jpg" for i in range(100)]
        detector.fit(small_train)

        # 测试
        small_test = [f"test_{i}.jpg" for i in range(20)]
        test_scores = detector.decision_function(small_test)

        print(f"{config['name']:15} - 平均分数: {test_scores.mean():.3f}, "
              f"标准差: {test_scores.std():.3f}")

    print("\n完成所有示例！")