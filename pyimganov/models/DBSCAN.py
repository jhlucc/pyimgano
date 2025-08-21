import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt


class DBSCANAnomalyDetector:
    """
    基于DBSCAN的图像异常检测器
    原理：DBSCAN将无法归入任何簇的点标记为噪声（异常）
    优点：
    1. 自动发现异常点
    2. 不需要指定簇数
    3. 能发现任意形状的簇
    """

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        """
        参数：
        eps: 邻域半径
        min_samples: 形成核心点的最小邻居数
        metric: 距离度量
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.dbscan = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=30)
        self.is_trained = False

    def extract_features(self, image_path):
        """提取图像特征"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        features = []

        # 1. 全局颜色特征
        for i in range(3):
            channel = img[:, :, i]
            # 颜色直方图
            hist, _ = np.histogram(channel, bins=16, range=(0, 256))
            hist = hist.astype(float) / (hist.sum() + 1e-6)
            features.extend(hist)

            # 统计特征
            features.extend([
                channel.mean() / 255.0,
                channel.std() / 255.0
            ])
            # 分别添加百分位数
            percentiles = np.percentile(channel, [10, 50, 90])
            features.extend(percentiles / 255.0)  # 归一化并展开添加

        # 2. HSV颜色空间特征（对光照更鲁棒）
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i in range(3):
            channel = hsv[:, :, i]
            features.extend([
                channel.mean() / 255.0,
                channel.std() / 255.0
            ])

        # 3. 纹理复杂度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 梯度特征
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        features.extend([
            grad_mag.mean() / 255.0,
            grad_mag.std() / 255.0,
            (grad_mag > 50).sum() / grad_mag.size  # 强边缘比例
        ])

        # 4. 频域特征（简化版）
        gray_small = cv2.resize(gray, (64, 64))
        fft = np.fft.fft2(gray_small)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # 低频和高频能量比
        center = magnitude.shape[0] // 2
        low_freq = magnitude[center - 8:center + 8, center - 8:center + 8].sum()
        high_freq = magnitude.sum() - low_freq
        features.append(low_freq / (high_freq + 1e-6))

        return np.array(features)

    def auto_select_eps(self, X, k=5):
        """自动选择eps参数（使用k-距离图）"""
        print(f"\n自动选择eps参数 (k={k})...")

        # 计算k-最近邻距离
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(X)
        distances, _ = nbrs.kneighbors(X)
        k_distances = distances[:, k - 1]  # 第k个邻居的距离

        # 排序
        k_distances_sorted = np.sort(k_distances)

        # 找肘部点（简单方法：最大曲率）
        # 计算二阶差分
        diffs = np.diff(k_distances_sorted)
        diffs2 = np.diff(diffs)

        # 找最大变化点
        if len(diffs2) > 0:
            elbow_idx = np.argmax(diffs2) + 2
            suggested_eps = k_distances_sorted[elbow_idx]
        else:
            suggested_eps = np.median(k_distances_sorted)

        # 绘制k-距离图
        plt.figure(figsize=(10, 6))
        plt.plot(k_distances_sorted)
        plt.axhline(y=suggested_eps, color='r', linestyle='--',
                    label=f'建议eps={suggested_eps:.3f}')
        plt.ylabel(f'{k}-距离')
        plt.xlabel('点（按距离排序）')
        plt.title('k-距离图')
        plt.legend()
        plt.grid(True)
        plt.show()

        return suggested_eps

    def train(self, data_folder, auto_eps=True):
        """训练DBSCAN模型"""
        print(f"开始训练DBSCAN异常检测器...")

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
        print(f"保留方差比例: {self.pca.explained_variance_ratio_.sum():.2%}")

        # 自动选择eps
        if auto_eps:
            self.eps = self.auto_select_eps(X_reduced, k=self.min_samples)

        # 训练DBSCAN
        print(f"\n使用参数: eps={self.eps:.3f}, min_samples={self.min_samples}")
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        self.dbscan.fit(X_reduced)

        # 分析结果
        labels = self.dbscan.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        print(f"\n聚类结果:")
        print(f"发现 {n_clusters} 个簇")
        print(f"噪声点（异常）: {n_noise}/{len(labels)} ({n_noise / len(labels) * 100:.1f}%)")

        # 统计每个簇的大小
        unique_labels = set(labels) - {-1}
        cluster_info = []
        for label in unique_labels:
            size = list(labels).count(label)
            cluster_info.append((label, size))

        cluster_info.sort(key=lambda x: x[1], reverse=True)
        print(f"\n簇大小分布:")
        for label, size in cluster_info[:5]:  # 显示前5个最大的簇
            print(f"  簇 {label}: {size} 个样本 ({size / len(labels) * 100:.1f}%)")

        # 输出噪声点
        if 0 < n_noise < 20:
            print(f"\n检测到的异常（噪声点）:")
            noise_indices = np.where(labels == -1)[0]
            for idx in noise_indices[:10]:  # 最多显示10个
                print(f"  - {filenames[idx]}")

        # 保存核心样本索引（用于预测）
        self.core_sample_indices_ = self.dbscan.core_sample_indices_
        self.X_train_scaled = X_reduced
        self.train_labels = labels
        self.is_trained = True

        return self

    def predict(self, image_path):
        """预测单张图片"""
        if not self.is_trained:
            raise ValueError("模型未训练！")

        # 提取特征
        feat = self.extract_features(image_path)
        feat_scaled = self.scaler.transform([feat])
        feat_reduced = self.pca.transform(feat_scaled)

        # 计算到所有核心点的最小距离
        if len(self.core_sample_indices_) > 0:
            core_samples = self.X_train_scaled[self.core_sample_indices_]
            distances = np.linalg.norm(core_samples - feat_reduced[0], axis=1)
            min_distance = np.min(distances)
            nearest_core_idx = self.core_sample_indices_[np.argmin(distances)]
            nearest_core_label = self.train_labels[nearest_core_idx]
        else:
            min_distance = np.inf
            nearest_core_label = -1

        # 判断是否在eps邻域内
        is_normal = min_distance <= self.eps

        # 计算异常分数
        anomaly_score = min_distance / self.eps

        # 找最近的k个训练样本
        distances_all = np.linalg.norm(self.X_train_scaled - feat_reduced[0], axis=1)
        k_nearest_indices = np.argsort(distances_all)[:self.min_samples]
        k_nearest_labels = self.train_labels[k_nearest_indices]

        # 邻域密度估计
        neighbors_in_eps = np.sum(distances_all <= self.eps)
        density_score = 1.0 - (neighbors_in_eps / self.min_samples)

        return {
            'image': os.path.basename(image_path),
            'is_normal': is_normal,
            'prediction': '正常' if is_normal else '异常（噪声点）',
            'min_distance_to_core': float(min_distance),
            'nearest_cluster': int(nearest_core_label) if nearest_core_label != -1 else '无',
            'anomaly_score': float(anomaly_score),
            'density_score': float(density_score),
            'neighbors_in_eps': int(neighbors_in_eps),
            'confidence': 1.0 - min(anomaly_score, 1.0) if is_normal else min(anomaly_score, 1.0)
        }

    def visualize_clustering(self):
        """可视化DBSCAN聚类结果"""
        if not self.is_trained:
            print("模型未训练！")
            return

        # 使用前两个主成分可视化
        X_2d = self.X_train_scaled[:, :2]

        plt.figure(figsize=(12, 5))

        # 子图1：聚类结果
        plt.subplot(1, 2, 1)

        # 获取唯一标签
        unique_labels = set(self.train_labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # 噪声点用黑色
                col = 'black'
                marker = 'x'
            else:
                marker = 'o'

            class_member_mask = (self.train_labels == k)
            xy = X_2d[class_member_mask]

            plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker,
                        s=50, label=f'簇 {k}' if k != -1 else '噪声',
                        edgecolors='black' if k == -1 else 'none',
                        alpha=0.8 if k != -1 else 1.0)

        plt.title('DBSCAN聚类结果')
        plt.xlabel('第一主成分')
        plt.ylabel('第二主成分')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 子图2：密度分析
        plt.subplot(1, 2, 2)

        # 计算每个点的局部密度
        nbrs = NearestNeighbors(radius=self.eps)
        nbrs.fit(self.X_train_scaled)

        densities = []
        for i in range(len(self.X_train_scaled)):
            neighbors = nbrs.radius_neighbors([self.X_train_scaled[i]], return_distance=False)[0]
            densities.append(len(neighbors))

        densities = np.array(densities)

        # 绘制密度分布
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=densities,
                              cmap='YlOrRd', s=50, alpha=0.8)
        plt.colorbar(scatter, label='邻域密度')
        plt.title('局部密度分布')
        plt.xlabel('第一主成分')
        plt.ylabel('第二主成分')

        plt.tight_layout()
        plt.show()

    def batch_predict(self, test_folder):
        """批量预测"""
        results = []

        print(f"\n批量预测 {test_folder} 中的图片...")

        for filename in tqdm(os.listdir(test_folder)):
            if filename.endswith('.jpg'):
                try:
                    img_path = os.path.join(test_folder, filename)
                    result = self.predict(img_path)
                    results.append(result)
                except Exception as e:
                    print(f"\n预测 {filename} 出错: {e}")

        # 统计
        n_normal = sum(1 for r in results if r['is_normal'])
        n_anomaly = len(results) - n_normal

        print(f"\n批量预测结果:")
        print(f"总数: {len(results)}")
        print(f"正常: {n_normal} ({n_normal / len(results) * 100:.1f}%)")
        print(f"异常: {n_anomaly} ({n_anomaly / len(results) * 100:.1f}%)")

        # 输出异常图片
        anomalies = [r for r in results if not r['is_normal']]
        if anomalies:
            print("\n检测到的异常图片（按异常分数排序）:")
            for r in sorted(anomalies, key=lambda x: x['anomaly_score'], reverse=True)[:10]:
                print(f"  - {r['image']}: 距离={r['min_distance_to_core']:.3f}, "
                      f"异常分数={r['anomaly_score']:.2f}")

        return results

    def save(self, filepath):
        """保存模型"""
        model_data = {
            'dbscan': self.dbscan,
            'scaler': self.scaler,
            'pca': self.pca,
            'eps': self.eps,
            'min_samples': self.min_samples,
            'core_sample_indices_': self.core_sample_indices_,
            'X_train_scaled': self.X_train_scaled,
            'train_labels': self.train_labels,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")


# 使用DBSCAN
if __name__ == "__main__":
    # 创建DBSCAN检测器
    detector = DBSCANAnomalyDetector(
        eps=0.5,  # 会自动调整
        min_samples=500,  # 至少500个邻居才能形成核心点
        metric='euclidean'
    )

    # 训练（自动选择eps）
    train_folder = "/data/temp7/程序正常"
    detector.train(train_folder, auto_eps=True)

    # 可视化聚类结果
    detector.visualize_clustering()

    # 批量测试
    results = detector.batch_predict(train_folder)

    # 测试单张图片
    test_image = "/data/temp11/程序正常/0a6c84ef7e1942529f46fea50ed5dfab.jpg"
    result = detector.predict(test_image)
    print(f"\nDBSCAN检测结果:")
    print(f"图片: {result['image']}")
    print(f"预测: {result['prediction']}")
    print(f"到最近核心点的距离: {result['min_distance_to_core']:.3f}")
    print(f"最近的簇: {result['nearest_cluster']}")
    print(f"异常分数: {result['anomaly_score']:.2f}")
    print(f"邻域内样本数: {result['neighbors_in_eps']}")

    # 保存模型
    detector.save("dbscan_detector.pkl")