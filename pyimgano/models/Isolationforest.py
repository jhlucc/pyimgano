import os
import cv2
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm

from .registry import register_model


@register_model(
    "isolation_forest_struct",
    tags=("vision", "classical", "ensemble"),
    metadata={"description": "结构特征 Isolation Forest 检测器"},
)
class IsolationForestStructureDetector:
    """
      结构异常检测器
    专注于：形状、纹理、边缘、布局等结构特征
    完全忽略颜色信息
    """

    def __init__(self, contamination=0.01, n_estimators=200, max_samples='auto'):
        """
        参数：
        contamination: 预期的异常比例
        n_estimators: 树的数量
        max_samples: 每棵树的样本数
        """
        self.iforest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=42,
            n_jobs=-1  # 并行处理
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=40)  # IF在中等维度效果好
        self.is_trained = False
        self.feature_dim = None  # 记录特征维度

    def extract_structure_features(self, image_path):
        """提取结构特征（形状、纹理、边缘）- 修复版"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 确保图像大小合理
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        features = []

        # 1. 形状特征 - 检测UI组件的形状
        # 自适应阈值分割（对UI元素效果好）
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # 连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # 组件统计
        component_areas = []
        component_aspects = []

        for i in range(1, num_labels):  # 跳过背景
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]

            if area > 50:  # 过滤小噪声
                component_areas.append(area)
                aspect_ratio = width / (height + 1e-6)
                component_aspects.append(aspect_ratio)

        features.extend([
            len(component_areas) / 100.0,  # 组件数量
            np.mean(component_areas) / (h * w) if component_areas else 0,
            np.std(component_areas) / (h * w) if component_areas else 0,
            np.mean(component_aspects) if component_aspects else 0,
            np.std(component_aspects) if component_aspects else 0
        ])

        # 2. 纹理特征 - 使用灰度共生矩阵（GLCM）的简化版
        # 计算局部对比度
        kernel_size = 3
        local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        local_sq_mean = cv2.blur((gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
        local_variance = np.maximum(local_sq_mean - local_mean ** 2, 0)  # 确保非负
        local_std = np.sqrt(local_variance)

        features.extend([
            local_std.mean() / 128.0,
            local_std.std() / 128.0,
            np.percentile(local_std, 10) / 128.0,
            np.percentile(local_std, 90) / 128.0
        ])

        # Gabor滤波器（检测不同方向和频率的纹理）
        gabor_features = []
        ksize = 31

        for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            for frequency in [0.05, 0.15]:
                try:
                    kernel = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, frequency, 0.5, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                    gabor_features.extend([
                        filtered.mean() / 255.0,
                        filtered.std() / 128.0
                    ])
                except Exception:
                    gabor_features.extend([0, 0])

        features.extend(gabor_features)

        # 3. 边缘和角点特征
        # 使用不同方法检测边缘
        edges_canny = cv2.Canny(gray, 50, 150)
        edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Harris角点检测
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corners_binary = corners > 0.01 * corners.max() if corners.max() > 0 else corners > 0

        features.extend([
            np.sum(edges_canny > 0) / edges_canny.size,  # Canny边缘密度
            np.abs(edges_laplacian).mean() / 255.0,  # Laplacian响应
            np.sum(corners_binary) / corners_binary.size * 1000  # 角点密度
        ])

        # 4. 结构复杂度特征
        # 使用形态学操作评估结构复杂度
        kernel_sizes = [3, 5, 7]
        morph_features = []

        for ksize in kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

            # 开运算（去除小物体）
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            diff_open = cv2.absdiff(binary, opened)

            # 闭运算（填充小洞）
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            diff_close = cv2.absdiff(closed, binary)

            morph_features.extend([
                np.sum(diff_open > 0) / diff_open.size,
                np.sum(diff_close > 0) / diff_close.size
            ])

        features.extend(morph_features)

        # 5. 布局特征（4x4网格）
        grid_size = 4
        cell_h, cell_w = max(1, h // grid_size), max(1, w // grid_size)

        # 计算每个网格的结构密度
        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * cell_h
                y_end = min((i + 1) * cell_h, h)
                x_start = j * cell_w
                x_end = min((j + 1) * cell_w, w)

                cell = edges_canny[y_start:y_end, x_start:x_end]
                density = np.sum(cell > 0) / cell.size if cell.size > 0 else 0
                features.append(density)

        # 6. 主导方向特征（UI通常有水平/垂直主导）
        # 使用Hough变换检测直线
        lines = cv2.HoughLinesP(edges_canny, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)

        angle_histogram = np.zeros(8)  # 8个方向

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                if dx != 0 or dy != 0:  # 避免零向量
                    angle = np.arctan2(dy, dx)
                    # 将角度映射到8个bin
                    bin_idx = int((angle + np.pi) / (np.pi / 4)) % 8
                    angle_histogram[bin_idx] += 1

        # 归一化
        if angle_histogram.sum() > 0:
            angle_histogram = angle_histogram / angle_histogram.sum()

        features.extend(angle_histogram.tolist())

        # 7. 分形维度（评估图像复杂度）- 修复数值稳定性
        # 使用盒计数法的简化版
        sizes = [2, 4, 8, 16]
        counts = []

        for size in sizes:
            # 将图像分成size x size的格子
            grid_h, grid_w = max(1, h // size), max(1, w // size)
            count = 0

            for i in range(size):
                for j in range(size):
                    y1 = i * grid_h
                    y2 = min((i + 1) * grid_h, h)
                    x1 = j * grid_w
                    x2 = min((j + 1) * grid_w, w)

                    if y2 > y1 and x2 > x1:
                        cell = edges_canny[y1:y2, x1:x2]
                        if np.any(cell > 0):
                            count += 1

            counts.append(max(1, count))  # 确保至少为1

        # 计算分形维度（斜率）
        if len(counts) > 1:
            try:
                log_sizes = np.log(sizes)
                log_counts = np.log(np.array(counts))
                # 使用线性回归计算斜率
                coeffs = np.polyfit(log_sizes, log_counts, 1)
                fractal_dim = abs(coeffs[0])
                features.append(np.clip(fractal_dim, 0, 3))  # 分形维度通常在0-3之间
            except Exception:
                features.append(1.5)  # 默认值
        else:
            features.append(1.5)

        # 确保所有特征都是有效数值
        features = [
            float(f) if not np.isnan(f) and not np.isinf(f) else 0.0
            for f in features
        ]

        return np.array(features)

    def train(self, data_folder, max_samples=None):
        """训练 Isolation Forest 模型"""
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"训练数据目录不存在: {data_folder}")

        print("开始训练 Isolation Forest 结构异常检测器...")
        print(f"参数: n_estimators={self.iforest.n_estimators}, "
              f"contamination={self.iforest.contamination}")

        features = []
        valid_files = []

        # 获取所有jpg文件
        jpg_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]

        if not jpg_files:
            raise ValueError(f"在 {data_folder} 中没有找到jpg文件")

        # 限制样本数量
        if max_samples and len(jpg_files) > max_samples:
            import random
            random.seed(42)  # 确保可重复性
            jpg_files = random.sample(jpg_files, max_samples)

        # 提取特征
        for filename in tqdm(jpg_files, desc="提取结构特征"):
            try:
                img_path = os.path.join(data_folder, filename)
                feat = self.extract_structure_features(img_path)
                features.append(feat)
                valid_files.append(filename)
            except Exception as e:
                print(f"\n处理 {filename} 出错: {e}")

        if not features:
            raise ValueError("没有成功提取任何特征")

        X = np.array(features)
        self.feature_dim = X.shape[1]  # 记录特征维度
        print(f"\n成功提取 {len(features)} 个样本的特征，原始维度: {X.shape[1]}")

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # PCA降维
        n_components = min(self.pca.n_components, X.shape[1] - 1, X.shape[0] - 1)
        self.pca.n_components = n_components
        X_reduced = self.pca.fit_transform(X_scaled)
        print(f"PCA降维后维度: {X_reduced.shape[1]}")
        print(f"保留方差比例: {self.pca.explained_variance_ratio_.sum():.2%}")

        # 输出前几个主成分的贡献度
        print("\n前10个主成分贡献度:")
        for i in range(min(10, len(self.pca.explained_variance_ratio_))):
            print(f"  PC{i + 1}: {self.pca.explained_variance_ratio_[i]:.2%}")

        # 训练 Isolation Forest
        print("\n训练 Isolation Forest...")
        self.iforest.fit(X_reduced)

        # 计算训练数据的异常分数
        self.train_scores = self.iforest.score_samples(X_reduced)
        self.train_predictions = self.iforest.predict(X_reduced)

        # 保存分数阈值
        self.score_threshold = np.percentile(self.train_scores, self.iforest.contamination * 100)

        # 添加额外的阈值用于分级
        self.score_threshold_high = np.percentile(self.train_scores, 1)  # 1%分位数
        self.score_threshold_medium = np.percentile(self.train_scores, 5)  # 5%分位数

        # 统计信息
        n_outliers = np.sum(self.train_predictions == -1)
        print(f"\n训练完成！")
        print(f"训练集中检测到的异常样本: {n_outliers}/{len(self.train_predictions)} "
              f"({n_outliers / len(self.train_predictions) * 100:.1f}%)")

        # 输出分数分布信息
        print(f"\n异常分数分布:")
        print(f"  最小值: {self.train_scores.min():.4f}")
        print(f"  最大值: {self.train_scores.max():.4f}")
        print(f"  均值: {self.train_scores.mean():.4f}")
        print(f"  标准差: {self.train_scores.std():.4f}")
        print(f"  决策阈值: {self.score_threshold:.4f}")
        print(f"  高异常阈值(1%): {self.score_threshold_high:.4f}")
        print(f"  中异常阈值(5%): {self.score_threshold_medium:.4f}")

        # 如果有异常样本，输出它们
        if 0 < n_outliers < 20:
            outlier_indices = np.where(self.train_predictions == -1)[0]
            print("\n训练集中的异常样本:")
            for idx in outlier_indices[:10]:  # 最多显示10个
                print(f"  - {valid_files[idx]} (异常分数: {self.train_scores[idx]:.3f})")

        self.is_trained = True
        return self

    def predict(self, image_path):
        """预测单张图片"""
        if not self.is_trained:
            raise ValueError("模型未训练！")

        # 提取特征
        feat = self.extract_structure_features(image_path)

        # 验证特征维度
        if len(feat) != self.feature_dim:
            raise ValueError(f"特征维度不匹配：期望{self.feature_dim}，实际{len(feat)}")

        feat_scaled = self.scaler.transform([feat])
        feat_reduced = self.pca.transform(feat_scaled)

        # 预测
        prediction = self.iforest.predict(feat_reduced)[0]
        anomaly_score = self.iforest.score_samples(feat_reduced)[0]

        # 计算相对异常程度 - 修正逻辑
        if anomaly_score < self.score_threshold_high:
            anomaly_level = "高异常"
        elif anomaly_score < self.score_threshold_medium:
            anomaly_level = "中异常"
        elif anomaly_score < self.score_threshold:
            anomaly_level = "轻微异常"
        else:
            anomaly_level = "正常"

        # 将分数转换为置信度（0-1）
        # IF的分数越低越异常，范围通常在[-0.5, 0.5]之间
        if prediction == 1:  # 正常
            # 正常样本的置信度：分数越高越确信是正常
            confidence = min(1.0, max(0, (anomaly_score - self.score_threshold) / 0.3))
        else:  # 异常
            # 异常样本的置信度：分数越低越确信是异常
            confidence = min(1.0, max(0, (self.score_threshold - anomaly_score) / 0.3))

        confidence = np.clip(confidence, 0, 1)

        return {
            'image': os.path.basename(image_path),
            'is_normal': prediction == 1,
            'prediction': '正常' if prediction == 1 else '异常',
            'anomaly_score': float(anomaly_score),
            'anomaly_level': anomaly_level,
            'confidence': float(confidence),
            'threshold': float(self.score_threshold),
            'score_percentile': float(np.sum(self.train_scores <= anomaly_score) / len(self.train_scores) * 100)
        }

    def batch_predict(self, test_folder):
        """批量预测"""
        if not os.path.exists(test_folder):
            raise FileNotFoundError(f"测试目录不存在: {test_folder}")

        results = []
        anomalies = []

        jpg_files = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]

        print(f"\n批量检测 {test_folder} 中的 {len(jpg_files)} 张图片...")

        for filename in tqdm(jpg_files, desc="检测进度"):
            try:
                img_path = os.path.join(test_folder, filename)
                result = self.predict(img_path)
                results.append(result)

                if not result['is_normal']:
                    anomalies.append(result)

            except Exception as e:
                print(f"\n检测 {filename} 出错: {e}")

        # 统计
        n_total = len(results)
        n_normal = sum(1 for r in results if r['is_normal'])
        n_anomaly = n_total - n_normal

        print(f"\n=== 检测结果统计 ===")
        print(f"总数: {n_total}")
        print(f"正常: {n_normal} ({n_normal / n_total * 100:.1f}%)")
        print(f"异常: {n_anomaly} ({n_anomaly / n_total * 100:.1f}%)")

        # 异常等级统计
        level_counts = {'高异常': 0, '中异常': 0, '轻微异常': 0}
        for r in results:
            level = r.get('anomaly_level', '正常')
            if level in level_counts:
                level_counts[level] += 1

        print(f"\n=== 异常等级分布 ===")
        for level, count in level_counts.items():
            print(f"{level}: {count} ({count / n_total * 100:.1f}%)")

        if anomalies:
            print(f"\n=== 检测到的异常（前10个）===")
            for a in sorted(anomalies, key=lambda x: x['anomaly_score'])[:10]:
                print(f"{a['image']}: {a['anomaly_level']} (异常分数: {a['anomaly_score']:.3f}, "
                      f"置信度: {a['confidence']:.2f}, 百分位: {a['score_percentile']:.1f}%)")

        return results

    def save(self, filepath):
        """保存模型"""
        model_data = {
            'iforest': self.iforest,
            'scaler': self.scaler,
            'pca': self.pca,
            'score_threshold': self.score_threshold,
            'score_threshold_high': getattr(self, 'score_threshold_high', self.score_threshold - 0.1),
            'score_threshold_medium': getattr(self, 'score_threshold_medium', self.score_threshold - 0.05),
            'train_scores': self.train_scores,
            'is_trained': self.is_trained,
            'feature_dim': self.feature_dim
        }
        joblib.dump(model_data, filepath)
        print(f"Isolation Forest 结构检测模型已保存到: {filepath}")

    def load(self, filepath):
        """加载模型"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        model_data = joblib.load(filepath)
        self.iforest = model_data['iforest']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.score_threshold = model_data['score_threshold']
        self.score_threshold_high = model_data.get('score_threshold_high', self.score_threshold - 0.1)
        self.score_threshold_medium = model_data.get('score_threshold_medium', self.score_threshold - 0.05)
        self.train_scores = model_data.get('train_scores', [])
        self.is_trained = model_data['is_trained']
        self.feature_dim = model_data.get('feature_dim', None)
        print(f"模型已从 {filepath} 加载")


# 使用示例
if __name__ == "__main__":
    # 创建检测器
    detector = IsolationForestStructureDetector(
        contamination=0.01,  # 1%的异常
        n_estimators=200,  # 200棵树
        max_samples='auto'
    )

    try:
        # 训练
        train_folder = "/data/temp7/程序正常"
        detector.train(train_folder, max_samples=40000)

        # 保存模型
        detector.save("iforest_structure_detector.pkl")

        # # 测试单张图片
        # test_image = "test.jpg"
        # if os.path.exists(test_image):
        #     result = detector.predict(test_image)
        #     print(f"\n单张图片检测结果:")
        #     print(f"  图片: {result['image']}")
        #     print(f"  预测: {result['prediction']}")
        #     print(f"  异常等级: {result['anomaly_level']}")
        #     print(f"  异常分数: {result['anomaly_score']:.3f}")
        #     print(f"  置信度: {result['confidence']:.2f}")
        #     print(f"  百分位: {result['score_percentile']:.1f}%")

    except Exception as e:
        print(f"运行出错: {e}")
        import traceback

        traceback.print_exc()
