import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class KMeansAnomalyDetector:
    """
    基于K-means的图像异常检测器
    原理：
    1. 距离簇中心太远的点是异常
    2. 属于很小簇的点可能是异常
    """

    def __init__(self, n_clusters=10, threshold_percentile=95, min_cluster_size_ratio=0.01):
        """
        参数：
        n_clusters: 簇的数量
        threshold_percentile: 距离阈值的百分位数
        min_cluster_size_ratio: 最小簇大小比例（小于此比例的簇中的点可能是异常）
        """
        self.n_clusters = n_clusters
        self.threshold_percentile = threshold_percentile
        self.min_cluster_size_ratio = min_cluster_size_ratio
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)
        self.is_trained = False

    def extract_features(self, image_path):
        """提取图像特征（复用之前的特征提取）"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        features = []

        # 1. 颜色直方图特征
        for i in range(3):
            channel = img[:, :, i]
            hist, _ = np.histogram(channel, bins=16, range=(0, 256))
            hist = hist.astype(float) / hist.sum()
            features.extend(hist)

        # 2. 颜色统计
        for i in range(3):
            channel = img[:, :, i]
            features.extend([
                channel.mean() / 255.0,
                channel.std() / 255.0,
                np.median(channel) / 255.0,
                np.percentile(channel, 25) / 255.0,
                np.percentile(channel, 75) / 255.0
            ])

        # 3. 空间颜色分布（3x3网格）
        h, w = img.shape[:2]
        grid_h, grid_w = h // 3, w // 3

        for i in range(3):
            for j in range(3):
                region = img[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
                features.extend([
                    region.mean() / 255.0,
                    region.std() / 255.0
                ])

        # 4. 纹理特征
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Laplacian（模糊度检测）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.append(laplacian.var() / 1000.0)  # 归一化

        # 边缘密度
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.sum(edges > 0) / edges.size)

        return np.array(features)

    def train(self, data_folder):
        """训练K-means模型"""
        print(f"开始训练K-means异常检测器 (n_clusters={self.n_clusters})...")

        features = []
        filenames = []

        # 提取特征
        jpg_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]

        for filename in tqdm(jpg_files, desc="提取特征"):
            try:
                img_path = os.path.join(data_folder, filename)
                feat = self.extract_features(img_path)
                features.append(feat)
                filenames.append(filename)
            except Exception as e:
                print(f"\n处理 {filename} 出错: {e}")

        X = np.array(features)
        print(f"\n成功提取 {len(features)} 个样本的特征，原始维度: {X.shape[1]}")

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # PCA降维
        X_reduced = self.pca.fit_transform(X_scaled)
        print(f"PCA降维后维度: {X_reduced.shape[1]}")

        # 训练K-means
        print("训练K-means...")
        self.kmeans.fit(X_reduced)

        # 计算每个点到其簇中心的距离
        distances = self._compute_distances_to_centers(X_reduced)

        # 设置距离阈值
        self.distance_threshold = np.percentile(distances, self.threshold_percentile)

        # 分析簇大小
        labels = self.kmeans.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        self.cluster_sizes = dict(zip(unique_labels, counts))
        self.min_cluster_size = int(len(X) * self.min_cluster_size_ratio)

        # 找出小簇
        small_clusters = [label for label, size in self.cluster_sizes.items()
                          if size < self.min_cluster_size]

        # 统计异常
        anomalies = []
        for i, (dist, label) in enumerate(zip(distances, labels)):
            if dist > self.distance_threshold or label in small_clusters:
                anomalies.append(i)

        print(f"\n训练完成！")
        print(f"簇大小分布: {sorted(counts)}")
        print(f"距离阈值: {self.distance_threshold:.3f}")
        print(f"小簇阈值: {self.min_cluster_size} 个样本")
        print(f"检测到异常: {len(anomalies)}/{len(X)} ({len(anomalies) / len(X) * 100:.1f}%)")

        if len(anomalies) > 0 and len(anomalies) < 10:
            print("\n异常样本:")
            for idx in anomalies[:5]:
                print(f"  - {filenames[idx]} (距离: {distances[idx]:.3f}, 簇: {labels[idx]})")

        self.is_trained = True
        self.X_train = X_reduced
        self.train_distances = distances
        self.train_labels = labels

        return self

    def _compute_distances_to_centers(self, X):
        """计算每个点到其所属簇中心的距离"""
        labels = self.kmeans.labels_
        centers = self.kmeans.cluster_centers_
        distances = np.zeros(len(X))

        for i in range(len(X)):
            distances[i] = np.linalg.norm(X[i] - centers[labels[i]])

        return distances

    def predict(self, image_path):
        """预测单张图片"""
        if not self.is_trained:
            raise ValueError("模型未训练！")

        # 提取特征
        feat = self.extract_features(image_path)
        feat_scaled = self.scaler.transform([feat])
        feat_reduced = self.pca.transform(feat_scaled)

        # 预测簇
        label = self.kmeans.predict(feat_reduced)[0]

        # 计算到簇中心的距离
        distance = np.linalg.norm(feat_reduced[0] - self.kmeans.cluster_centers_[label])

        # 判断是否异常
        is_distance_anomaly = distance > self.distance_threshold
        is_small_cluster = self.cluster_sizes[label] < self.min_cluster_size
        is_anomaly = is_distance_anomaly or is_small_cluster

        # 计算异常分数
        distance_score = distance / self.distance_threshold
        cluster_score = 1.0 - (self.cluster_sizes[label] / max(self.cluster_sizes.values()))
        anomaly_score = max(distance_score, cluster_score)

        # 异常原因
        anomaly_reasons = []
        if is_distance_anomaly:
            anomaly_reasons.append(f"距离异常 (距离={distance:.3f} > 阈值={self.distance_threshold:.3f})")
        if is_small_cluster:
            anomaly_reasons.append(f"小簇异常 (簇大小={self.cluster_sizes[label]} < {self.min_cluster_size})")

        return {
            'image': os.path.basename(image_path),
            'is_normal': not is_anomaly,
            'prediction': '正常' if not is_anomaly else '异常',
            'cluster_label': int(label),
            'distance_to_center': float(distance),
            'cluster_size': self.cluster_sizes[label],
            'anomaly_score': float(anomaly_score),
            'anomaly_reasons': anomaly_reasons,
            'confidence': min(anomaly_score, 1.0) if is_anomaly else 1.0 - anomaly_score
        }

    def visualize_clusters(self):
        """可视化聚类结果（使用前两个主成分）"""
        if not self.is_trained:
            print("模型未训练！")
            return

        plt.figure(figsize=(12, 5))

        # 子图1：聚类结果
        plt.subplot(1, 2, 1)

        # 绘制数据点
        scatter = plt.scatter(self.X_train[:, 0], self.X_train[:, 1],
                              c=self.train_labels, cmap='tab10',
                              alpha=0.6, s=50)

        # 绘制簇中心
        centers_2d = self.kmeans.cluster_centers_[:, :2]
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1],
                    c='red', marker='x', s=200, linewidths=3)

        plt.colorbar(scatter)
        plt.title('K-means聚类结果')
        plt.xlabel('第一主成分')
        plt.ylabel('第二主成分')

        # 子图2：距离分布
        plt.subplot(1, 2, 2)
        plt.hist(self.train_distances, bins=50, alpha=0.7, color='blue')
        plt.axvline(x=self.distance_threshold, color='red', linestyle='--',
                    label=f'阈值: {self.distance_threshold:.3f}')
        plt.xlabel('到簇中心的距离')
        plt.ylabel('频数')
        plt.title('距离分布')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def find_optimal_k(self, data_folder, k_range=range(2, 21)):
        """使用肘部法则找最优的k值"""
        print("寻找最优的簇数量k...")

        # 提取特征（复用训练的特征提取）
        features = []
        jpg_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')][:100]  # 只用前100张

        for filename in tqdm(jpg_files, desc="提取特征"):
            try:
                img_path = os.path.join(data_folder, filename)
                feat = self.extract_features(img_path)
                features.append(feat)
            except:
                pass

        X = np.array(features)
        X_scaled = self.scaler.fit_transform(X)
        X_reduced = self.pca.fit_transform(X_scaled)

        # 计算不同k值的SSE
        sse = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans.fit(X_reduced)
            sse.append(kmeans.inertia_)

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, sse, 'bo-')
        plt.xlabel('簇数量 k')
        plt.ylabel('SSE (簇内平方和)')
        plt.title('肘部法则：寻找最优k值')
        plt.grid(True)

        # 标记可能的肘部点
        if len(sse) > 2:
            # 计算二阶差分找肘部
            diffs = np.diff(sse)
            diffs2 = np.diff(diffs)
            elbow_idx = np.argmax(diffs2) + 2
            optimal_k = list(k_range)[elbow_idx]
            plt.axvline(x=optimal_k, color='red', linestyle='--',
                        label=f'建议k={optimal_k}')
            plt.legend()

        plt.show()


# 使用K-means
if __name__ == "__main__":
    # 先找最优k值
    detector = KMeansAnomalyDetector()
    detector.find_optimal_k("/Computer/data/temp11/程序正常")

    # 创建检测器
    detector = KMeansAnomalyDetector(
        n_clusters=10,  # 根据肘部法则选择
        threshold_percentile=95,
        min_cluster_size_ratio=0.02  # 小于2%的簇可能是异常
    )

    # 训练
    detector.train("/Computer/data/temp11/程序正常")

    # 可视化
    detector.visualize_clusters()

    # 测试
    test_image = "/Computer/data/temp11/程序正常/0a6c84ef7e1942529f46fea50ed5dfab.jpg"
    result = detector.predict(test_image)
    print(f"\nK-means检测结果: {result}")

#数据有明显的簇结构 需要设置阈值 球形簇