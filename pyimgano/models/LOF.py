import os
import cv2
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm


class LOFStructureAnomalyDetector:
    """
    基于LOF的结构异常检测器
    专注于：边缘、形状、纹理、布局等结构特征
    忽略颜色信息
    """

    def __init__(self, n_neighbors=50, contamination=0.01, leaf_size=30):
        """
        参数：
        n_neighbors: 考虑的邻居数量
        contamination: 预期的异常比例
        leaf_size: 构建树的叶子大小，影响速度
        """
        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,  # 允许预测新数据
            leaf_size=leaf_size,
            metric='minkowski',
            p=2
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)  # 结构特征可能需要更多维度
        self.is_trained = False
        self.feature_dim = None  # 记录特征维度

    def calculate_symmetry(self, gray):
        """对称性计算方法"""
        h, w = gray.shape

        # 水平对称（左右）
        half_w = w // 2
        if half_w > 0:
            left_half = gray[:, :half_w]
            right_half = gray[:, -half_w:]  # 从右边取相同大小
            if left_half.shape == right_half.shape:
                right_flipped = cv2.flip(right_half, 1)
                h_symmetry = 1 - np.mean(np.abs(left_half.astype(float) - right_flipped.astype(float))) / 255.0
            else:
                h_symmetry = 0.5
        else:
            h_symmetry = 0.5

        # 垂直对称（上下）
        half_h = h // 2
        if half_h > 0:
            top_half = gray[:half_h, :]
            bottom_half = gray[-half_h:, :]  # 从底部取相同大小
            if top_half.shape == bottom_half.shape:
                bottom_flipped = cv2.flip(bottom_half, 0)
                v_symmetry = 1 - np.mean(np.abs(top_half.astype(float) - bottom_flipped.astype(float))) / 255.0
            else:
                v_symmetry = 0.5
        else:
            v_symmetry = 0.5

        return h_symmetry, v_symmetry

    def extract_structure_features(self, image_path):
        """优化后的结构特征提取，去掉耗时操作"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        max_dim = 512  # 将大图缩小到512px以内
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        features = []

        # 1. 边缘特征
        edges_low = cv2.Canny(gray, 50, 100)
        edges_high = cv2.Canny(gray, 100, 200)

        # 边缘密度
        features.extend([
            np.sum(edges_low > 0) / edges_low.size,
            np.sum(edges_high > 0) / edges_high.size
        ])

        # 2. 梯度特征
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 梯度统计
        features.extend([
            grad_mag.mean() / 255.0,
            grad_mag.std() / 255.0,
            np.percentile(grad_mag, 90) / 255.0  # 强边缘
        ])

        # 3. 简化的纹理特征
        ws = 16  # 只用16x16窗口
        stride = ws  # 步长等于窗口大小，不重叠
        local_vars = []

        for i in range(0, h - ws + 1, stride):
            for j in range(0, w - ws + 1, stride):
                window = gray[i:i + ws, j:j + ws]
                local_vars.append(np.var(window))

        if local_vars:
            features.extend([
                np.mean(local_vars) / 1000,
                np.std(local_vars) / 1000,
                np.percentile(local_vars, 90) / 1000
            ])
        else:
            features.extend([0, 0, 0])
        # 4. 形状特征
        # 检测水平和垂直结构
        h_kernel = np.ones((1, 20), np.uint8)
        v_kernel = np.ones((20, 1), np.uint8)

        h_lines = cv2.morphologyEx(edges_high, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(edges_high, cv2.MORPH_OPEN, v_kernel)

        h_density = np.sum(h_lines > 0) / h_lines.size
        v_density = np.sum(v_lines > 0) / v_lines.size

        # 对角线检测
        d1_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        d1_kernel = np.eye(20, dtype=np.uint8)
        d2_kernel = np.fliplr(d1_kernel)

        d1_lines = cv2.morphologyEx(edges_high, cv2.MORPH_OPEN, d1_kernel)
        d2_lines = cv2.morphologyEx(edges_high, cv2.MORPH_OPEN, d2_kernel)

        d1_density = np.sum(d1_lines > 0) / d1_lines.size
        d2_density = np.sum(d2_lines > 0) / d2_lines.size

        features.extend([
            h_density * 100,  # 水平线密度
            v_density * 100,  # 垂直线密度
            (d1_density + d2_density) * 50,  # 对角线密度
            (h_density + v_density) / (0.01 + d1_density + d2_density)  # 结构化程度
        ])

        # 5. 轮廓特征
        contours, _ = cv2.findContours(edges_high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 只统计大轮廓
        large_contours = [c for c in contours if cv2.contourArea(c) > 100]
        # 检测矩形轮廓
        rect_count = 0
        for contour in large_contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:  # 四边形
                rect_count += 1

        features.extend([
            len(large_contours) / 100.0,  # 轮廓数量
            len(large_contours) / (h * w) * 10000,  # 轮廓密度
            rect_count / (len(large_contours) + 1)  # 矩形比例
        ])

        # 6. 区域布局特征（改为2x2网格）
        grid_size = 2  # 从3x3改为2x2
        cell_h, cell_w = h // grid_size, w // grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * cell_h
                y_end = (i + 1) * cell_h if i < grid_size - 1 else h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w if j < grid_size - 1 else w

                cell = edges_high[y_start:y_end, x_start:x_end]
                density = np.sum(cell > 0) / cell.size if cell.size > 0 else 0
                features.append(density)

        # 7. 对称性计算
        h_symmetry, v_symmetry = self.calculate_symmetry(gray)
        features.extend([h_symmetry, v_symmetry])

        # 8. 添加中心偏向性特征（替代硬编码的0.5）
        # 计算边缘的中心偏向性
        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.where(edges_high > 0)
        if len(y_coords) > 0:
            # 边缘点到中心的平均距离
            distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
            max_dist = np.sqrt(center_y ** 2 + center_x ** 2)
            center_bias = 1 - (distances.mean() / max_dist)
            edge_spread = distances.std() / max_dist
        else:
            center_bias = 0.5
            edge_spread = 0.5

        features.extend([center_bias, edge_spread])

        # 确保所有特征都是有效数值
        features = [float(f) if not np.isnan(f) and not np.isinf(f) else 0.0 for f in features]

        return np.array(features)

    def train(self, data_folder, max_samples=None):
        """训练LOF模型"""
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"训练数据目录不存在: {data_folder}")

        print("开始训练LOF结构异常检测器...")
        print(f"参数: n_neighbors={self.lof.n_neighbors}, contamination={self.lof.contamination}")

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

        # 训练LOF
        print("训练LOF模型...")
        self.lof.fit(X_reduced)

        # 计算训练数据的LOF分数
        self.train_lof_scores = self.lof.score_samples(X_reduced)
        self.train_predictions = self.lof.predict(X_reduced)

        # 统计信息
        n_outliers = np.sum(self.train_predictions == -1)
        print(f"\n训练完成！")
        print(f"训练集中检测到的异常样本: {n_outliers}/{len(self.train_predictions)} "
              f"({n_outliers / len(self.train_predictions) * 100:.1f}%)")

        # 保存阈值信息
        self.score_threshold_high = np.percentile(self.train_lof_scores, 95)
        self.score_threshold_low = np.percentile(self.train_lof_scores, 5)
        self.offset_ = self.lof.offset_


        # 如果有异常样本，输出它们
        if 0 < n_outliers < 20:
            outlier_indices = np.where(self.train_predictions == -1)[0]
            print("\n训练集中的异常样本:")
            for idx in outlier_indices[:10]:  # 最多显示10个
                print(f"  - {valid_files[idx]} (LOF分数: {self.train_lof_scores[idx]:.3f})")

        self.is_trained = True
        self.X_train = X_reduced  # 保存训练数据用于可视化

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

        # LOF预测
        prediction = self.lof.predict(feat_reduced)[0]
        lof_score = self.lof.score_samples(feat_reduced)[0]

        # 找出最近的k个邻居
        distances, indices = self.lof.kneighbors(feat_reduced)
        k_distance = distances[0][-1]  # 第k个邻居的距离
        avg_neighbor_distance = distances[0].mean()

        # 计算相对异常程度
        if lof_score < self.score_threshold_low:
            anomaly_level = "高异常"
        elif lof_score < self.offset_:
            anomaly_level = "中异常"
        else:
            anomaly_level = "正常"

        # 置信度计算
        if prediction == 1:  # 正常
            if self.score_threshold_high > self.offset_:
                confidence = min(1.0, (lof_score - self.offset_) / (self.score_threshold_high - self.offset_))
            else:
                confidence = 0.5
        else:  # 异常
            if self.offset_ > self.score_threshold_low:
                confidence = min(1.0, (self.offset_ - lof_score) / abs(self.score_threshold_low - self.offset_))
            else:
                confidence = 0.5

        confidence = np.clip(confidence, 0, 1)

        return {
            'image': os.path.basename(image_path),
            'is_normal': prediction == 1,
            'prediction': '正常' if prediction == 1 else '异常',
            'lof_score': float(lof_score),
            'anomaly_level': anomaly_level,
            'confidence': float(confidence),
            'k_distance': float(k_distance),
            'avg_neighbor_distance': float(avg_neighbor_distance),
            'decision_boundary': float(self.offset_)
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

        if anomalies:
            print(f"\n=== 检测到的异常（前10个）===")
            for a in sorted(anomalies, key=lambda x: x['lof_score'])[:10]:
                print(f"{a['image']}: {a['anomaly_level']} (LOF分数: {a['lof_score']:.3f}, "
                      f"置信度: {a['confidence']:.2f})")

        return results

    def save(self, filepath):
        """保存模型"""
        model_data = {
            'lof': self.lof,
            'scaler': self.scaler,
            'pca': self.pca,
            'score_threshold_high': self.score_threshold_high,
            'score_threshold_low': self.score_threshold_low,
            'offset_': self.offset_,
            'is_trained': self.is_trained,
            'feature_dim': self.feature_dim
        }
        joblib.dump(model_data, filepath)
        print(f"LOF结构检测模型已保存到: {filepath}")

    def load(self, filepath):
        """加载模型"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        model_data = joblib.load(filepath)
        self.lof = model_data['lof']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.score_threshold_high = model_data['score_threshold_high']
        self.score_threshold_low = model_data['score_threshold_low']
        self.offset_ = model_data['offset_']
        self.is_trained = model_data['is_trained']
        self.feature_dim = model_data.get('feature_dim', None)
        print(f"模型已从 {filepath} 加载")


# 使用示例
if __name__ == "__main__":
    # 创建LOF检测器
    detector = LOFStructureAnomalyDetector(
        n_neighbors=100,  # 增加邻居数
        contamination=0.01,  # 1%的异常
        leaf_size=30
    )

    try:
        # 训练
        train_folder = "/data/temp7/程序正常"
        detector.train(train_folder, max_samples=70000)  # 限制样本数量

        # 保存模型
        detector.save("lof_structure_detector.pkl")

        # # 测试单张图片
        # test_image = "test.jpg"
        # if os.path.exists(test_image):
        #     result = detector.predict(test_image)
        #     print(f"\n单张图片检测结果:")
        #     print(f"  图片: {result['image']}")
        #     print(f"  预测: {result['prediction']}")
        #     print(f"  异常等级: {result['anomaly_level']}")
        #     print(f"  LOF分数: {result['lof_score']:.3f}")
        #     print(f"  置信度: {result['confidence']:.2f}")

    except Exception as e:
        print(f"运行出错: {e}")
        import traceback

        traceback.print_exc()