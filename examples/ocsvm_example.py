import os
import cv2
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm
import random

class StructureAnomalyDetector:
    """
    结构异常检测器 - 专注于弹窗、白屏、黑屏、遮挡等
    忽略颜色变化，只关注结构性异常
    """
    def __init__(self, nu=0.01, kernel='rbf', max_samples=2000):
        self.nu = nu
        self.kernel = kernel
        self.max_samples = max_samples
        self.ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma='auto')
        self.scaler = StandardScaler()
        self.pca = None
        self.is_trained = False

        # 定义忽略区域（时间等动态内容）
        self.ignore_regions = [
            {"x": 80, "y": 55, "w": 200, "h": 20},  # 时间区域
        ]

    def create_mask(self, shape, scale=1.0):
        """创建忽略区域的掩码"""
        h, w = int(shape[0] * scale), int(shape[1] * scale)
        mask = np.ones((h, w), dtype=np.uint8) * 255

        for region in self.ignore_regions:
            x = int(region['x'] * scale)
            y = int(region['y'] * scale)
            w = int(region['w'] * scale)
            h = int(region['h'] * scale)
            mask[y:y + h, x:x + w] = 0

        return mask

    def detect_white_black_screen(self, gray_img, mask):
        """检测白屏或黑屏"""
        valid_pixels = gray_img[mask > 0]
        if len(valid_pixels) == 0:
            return 0, 0

        # 白屏：大部分像素接近255
        white_ratio = np.sum(valid_pixels > 250) / len(valid_pixels)

        # 黑屏：大部分像素接近0
        black_ratio = np.sum(valid_pixels < 5) / len(valid_pixels)

        return white_ratio, black_ratio

    def detect_popup_features(self, gray_img):
        """检测弹窗特征"""
        # 边缘检测
        edges = cv2.Canny(gray_img, 100, 200)

        # 检测直线（弹窗通常有明显的直线边框）
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=10)

        if lines is None:
            return 0, 0, 0

        # 统计水平和垂直线
        horizontal_lines = 0
        vertical_lines = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1))

            if angle < np.pi / 6 or angle > 5 * np.pi / 6:  # 接近水平
                horizontal_lines += 1
            elif np.pi / 3 < angle < 2 * np.pi / 3:  # 接近垂直
                vertical_lines += 1

        # 检测矩形（弹窗通常是矩形）
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = 0

        for contour in contours:
            # 拟合矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # 计算矩形度
            area = cv2.contourArea(contour)
            rect_area = rect[1][0] * rect[1][1]

            if rect_area > 0 and area / rect_area > 0.8:  # 高矩形度
                if rect[1][0] > 50 and rect[1][1] > 50:  # 足够大
                    rectangles += 1

        return horizontal_lines, vertical_lines, rectangles

    def extract_features(self, image_path):
        """提取结构性特征，忽略颜色"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 统一尺寸
        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = self.create_mask(img.shape)

        features = []

        # 1. 白屏/黑屏检测
        white_ratio, black_ratio = self.detect_white_black_screen(gray, mask)
        features.extend([white_ratio, black_ratio])

        # 2. 弹窗检测特征
        h_lines, v_lines, rectangles = self.detect_popup_features(gray)
        features.extend([
            h_lines / 100.0,  # 归一化
            v_lines / 100.0,
            rectangles / 10.0,
            (h_lines + v_lines) / 200.0  # 总直线数
        ])

        # 3. 边缘分布特征（结构性）
        edges = cv2.Canny(gray, 50, 150)

        # 将图像分成3x3网格，计算每个区域的边缘密度
        h, w = edges.shape
        grid_h, grid_w = h // 3, w // 3

        edge_distribution = []
        for i in range(3):
            for j in range(3):
                region = edges[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
                region_mask = mask[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]

                if np.sum(region_mask > 0) > 0:
                    edge_density = np.sum(region[region_mask > 0] > 0) / np.sum(region_mask > 0)
                else:
                    edge_density = 0

                edge_distribution.append(edge_density)

        features.extend(edge_distribution)

        # 4. 梯度方向直方图（HOG-like，检测形状）
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # 计算梯度幅值和方向
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        angle = np.arctan2(gy, gx)

        # 8个方向的直方图
        hist_bins = 8
        angle_hist, _ = np.histogram(angle[mask > 0], bins=hist_bins, range=(-np.pi, np.pi))
        angle_hist = angle_hist / (np.sum(angle_hist) + 1e-6)
        features.extend(angle_hist)

        # 5. 纹理复杂度（用于检测键盘等复杂结构）
        # 局部二值模式（LBP）简化版
        texture_features = []
        for i in range(2):
            for j in range(2):
                region = gray[i * 240:(i + 1) * 240, j * 320:(j + 1) * 320]
                # 计算局部方差
                local_var = np.var(region)
                texture_features.append(local_var / 10000.0)  # 归一化

        features.extend(texture_features)

        # 6. 连通组件分析（检测独立元素数量）
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, _ = cv2.connectedComponents(binary)
        features.append(num_labels / 100.0)  # 归一化

        # 7. 对称性检测（正常界面通常有一定对称性）
        mid = w // 2
        left_half = gray[:, :mid]
        right_half = cv2.flip(gray[:, mid:], 1)

        if left_half.shape == right_half.shape:
            symmetry_score = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
        else:
            symmetry_score = 0
        features.append(symmetry_score)

        return np.array(features)

    def train(self, data_folder):
        """训练模型"""
        print(f"开始训练结构异常检测器...")
        print(f"参数: nu={self.nu}, kernel={self.kernel}")
        print("注意：只检测结构异常（弹窗、白屏、黑屏、遮挡），忽略颜色差异")

        features = []
        valid_files = []

        # 获取所有jpg文件
        jpg_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]

        # 如果文件太多，随机采样
        if len(jpg_files) > self.max_samples:
            print(f"文件数量({len(jpg_files)})超过限制({self.max_samples})，随机采样...")
            jpg_files = random.sample(jpg_files, self.max_samples)

        # 提取特征
        for filename in tqdm(jpg_files, desc="提取结构特征"):
            try:
                img_path = os.path.join(data_folder, filename)
                feat = self.extract_features(img_path)
                features.append(feat)
                valid_files.append(filename)
            except Exception as e:
                print(f"\n处理 {filename} 出错: {e}")

        X = np.array(features)
        print(f"\n成功提取 {len(features)} 个样本的特征，原始维度: {X.shape[1]}")

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # PCA降维（动态设置维度）
        n_components = min(20, X.shape[1] - 1, X.shape[0] - 1)
        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X_scaled)
        print(f"PCA降维后维度: {X_reduced.shape[1]}")
        print(f"保留方差比例: {self.pca.explained_variance_ratio_.sum():.2%}")

        # 训练One-Class SVM
        print("训练One-Class SVM...")
        self.ocsvm.fit(X_reduced)

        # 计算训练集上的决策边界
        self.decision_scores = self.ocsvm.decision_function(X_reduced)
        self.threshold_normal = np.percentile(self.decision_scores, 99)
        self.threshold_anomaly = np.percentile(self.decision_scores, 1)

        self.is_trained = True
        print("训练完成！")

        # 输出训练统计
        predictions = self.ocsvm.predict(X_reduced)
        n_outliers = np.sum(predictions == -1)
        print(f"训练集中检测到的结构异常样本: {n_outliers}/{len(predictions)} "
              f"({n_outliers / len(predictions) * 100:.1f}%)")

        return self

    def predict(self, image_path):
        """预测单张图片"""
        if not self.is_trained:
            raise ValueError("模型未训练！")

        # 提取特征
        feat = self.extract_features(image_path)
        feat_scaled = self.scaler.transform([feat])
        feat_reduced = self.pca.transform(feat_scaled)

        # 预测
        prediction = self.ocsvm.predict(feat_reduced)[0]
        decision_score = self.ocsvm.decision_function(feat_reduced)[0]

        # 分析异常原因（基于结构特征）
        anomaly_type = "未知"
        if prediction == -1:
            # 检查具体是哪种结构异常
            if feat[0] > 0.9:  # 白屏
                anomaly_type = "白屏"
            elif feat[1] > 0.9:  # 黑屏
                anomaly_type = "黑屏"
            elif feat[4] > 0.5:  # 检测到多个矩形
                anomaly_type = "可能有弹窗"
            elif feat[2] > 0.5 or feat[3] > 0.5:  # 很多直线
                anomaly_type = "界面结构异常"
            else:
                # 检查边缘分布
                edge_features = feat[6:15]
                if np.std(edge_features) > 0.3:
                    anomaly_type = "布局异常"

        # 计算置信度
        if prediction == 1:
            confidence = min((decision_score - self.threshold_anomaly) /
                             (self.threshold_normal - self.threshold_anomaly), 1.0)
        else:
            confidence = min((self.threshold_anomaly - decision_score) /
                             abs(self.threshold_anomaly), 1.0)

        confidence = np.clip(confidence, 0, 1)

        return {
            'image': os.path.basename(image_path),
            'is_normal': prediction == 1,
            'prediction': '正常' if prediction == 1 else '异常',
            'decision_score': float(decision_score),
            'confidence': float(confidence),
            'anomaly_type': anomaly_type if prediction == -1 else None,
            'structure_features': {
                'white_screen_ratio': float(feat[0]),
                'black_screen_ratio': float(feat[1]),
                'horizontal_lines': int(feat[2] * 100),
                'vertical_lines': int(feat[3] * 100),
                'rectangles_detected': int(feat[4] * 10)
            }
        }

    def save(self, filepath):
        """保存模型"""
        model_data = {
            'ocsvm': self.ocsvm,
            'scaler': self.scaler,
            'pca': self.pca,
            'threshold_normal': self.threshold_normal,
            'threshold_anomaly': self.threshold_anomaly,
            'is_trained': self.is_trained,
            'nu': self.nu,
            'kernel': self.kernel
        }
        joblib.dump(model_data, filepath)
        print(f"结构异常检测模型已保存到: {filepath}")


# 使用示例
if __name__ == "__main__":
    # 结构异常的检测器
    detector = StructureAnomalyDetector(
        nu=0.001,
        kernel='rbf',
        max_samples=50000
    )

    # 训练
    detector.train("/data/temp7/程序正常")

    # 保存模型
    detector.save("svm_anomaly_detector.pkl")

    # 测试那张被误判为颜色异常的图片
    result = detector.predict("/data/temp7/2/0003e776f766442592f074fd9262e52b.jpg")
    print("\n结构异常检测结果:")
    print(result)