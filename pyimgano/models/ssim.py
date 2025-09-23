import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
import joblib
from datetime import datetime
import json

from .registry import register_model


@register_model(
    "ssim_template",
    tags=("vision", "classical", "template"),
    metadata={"description": "基于模板匹配的 SSIM 异常检测器"},
)
class OneStopServiceAnomalyDetector:
    """
    一体机自助服务界面异常检测器
    专门检测：弹窗、错误提示、界面遮挡、异常状态
    """

    def __init__(self, n_templates=10, similarity_threshold=0.90):
        self.n_templates = n_templates  # 增加模板数量以覆盖不同时间
        self.similarity_threshold = similarity_threshold  # 提高阈值，因为界面很固定
        self.templates = []
        self.template_times = []  # 记录模板的时间，用于匹配
        self.is_trained = False

        # 定义需要忽略的动态区域（根据你的截图调整）
        self.ignore_regions = [
            # 时间显示区域 - 根据你的截图，时间在左上角
            {"name": "datetime", "x": 80, "y": 55, "w": 200, "h": 20},
            # 可能的其他动态区域
        ]

        # 定义关键检测区域（功能按钮等）
        self.key_regions = [
            {"name": "main_buttons", "x": 50, "y": 150, "w": 580, "h": 400},
            {"name": "bottom_area", "x": 0, "y": 700, "w": 1000, "h": 100}
        ]

    def create_mask(self, img_shape):
        """创建忽略动态区域的掩码"""
        mask = np.ones(img_shape[:2], dtype=np.uint8) * 255

        for region in self.ignore_regions:
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            mask[y:y + h, x:x + w] = 0

        return mask

    def resize_image(self, img, target_size=(512, 384)):
        """统一图片尺寸 - 适配一体机分辨率"""
        return cv2.resize(img, target_size)

    def compute_masked_ssim(self, img1, img2, mask=None):
        """计算忽略特定区域的SSIM"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if mask is not None:
            # 将忽略区域设为相同值
            gray1_masked = gray1.copy()
            gray2_masked = gray2.copy()
            gray1_masked[mask == 0] = 128
            gray2_masked[mask == 0] = 128
            score, diff = ssim(gray1_masked, gray2_masked, full=True)
        else:
            score, diff = ssim(gray1, gray2, full=True)

        return score, diff

    def detect_popup_window(self, img):
        """专门检测弹窗"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. 边缘检测
        edges = cv2.Canny(gray, 100, 200)

        # 2. 形态学操作，连接边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 3. 查找轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        popup_candidates = []
        img_h, img_w = img.shape[:2]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # 弹窗的特征：
            # - 通常是矩形
            # - 在屏幕中央区域
            # - 有一定的大小
            # - 宽高比合理

            # 检查大小
            if w < 100 or h < 50:  # 太小
                continue
            if w > img_w * 0.9 or h > img_h * 0.9:  # 太大
                continue

            # 检查位置（通常在中央）
            center_x = x + w / 2
            center_y = y + h / 2
            if not (img_w * 0.2 < center_x < img_w * 0.8):
                continue
            if not (img_h * 0.2 < center_y < img_h * 0.8):
                continue

            # 检查矩形度
            rect_area = w * h
            contour_area = cv2.contourArea(contour)
            if contour_area / rect_area < 0.8:  # 不够矩形
                continue

            popup_candidates.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'score': w * h,  # 用面积作为分数
                'center': (center_x, center_y)
            })

        # 返回最可能的弹窗
        if popup_candidates:
            popup_candidates.sort(key=lambda x: x['score'], reverse=True)
            return True, popup_candidates[0]

        return False, None

    def extract_time_features(self, img):
        """提取时间特征用于模板匹配"""
        # 简单提取：一天中的时段（早中晚）
        # 实际可以通过OCR读取时间
        return {
            'brightness': np.mean(img),  # 用亮度简单区分
            'color_tone': np.mean(img, axis=(0, 1))  # 色调
        }

    def train(self, data_folder):
        """训练：选择代表性模板"""
        print(f"开始训练一体机界面异常检测器...")
        print(f"数据目录: {data_folder}")

        images = []
        features = []
        filenames = []

        # 读取所有图片
        jpg_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]
        print(f"找到 {len(jpg_files)} 张图片")

        for filename in jpg_files[:200]:  # 限制数量避免内存过大
            img_path = os.path.join(data_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = self.resize_image(img)
                images.append(img_resized)

                # 提取聚类特征（降采样+掩码）
                mask = self.create_mask(img_resized.shape)
                small = cv2.resize(img_resized, (64, 64))
                small_mask = cv2.resize(mask, (64, 64))

                # 只使用非忽略区域的特征
                masked_pixels = small[small_mask > 0]

                # 使用颜色直方图作为特征
                hist_features = []
                for i in range(3):
                    hist = cv2.calcHist([small], [i], small_mask, [16], [0, 256])
                    hist_features.extend(hist.flatten())

                features.append(hist_features)
                filenames.append(filename)

        print(f"成功读取 {len(images)} 张图片")

        # 使用K-means选择代表性模板
        X = np.array(features)

        # 动态调整簇数量
        actual_n_templates = min(self.n_templates, len(images))
        kmeans = KMeans(n_clusters=actual_n_templates, random_state=42, n_init=10)
        kmeans.fit(X)

        # 选择每个簇的代表
        selected_indices = []
        for i in range(actual_n_templates):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            if len(cluster_indices) > 0:
                # 找到离中心最近的样本
                center = kmeans.cluster_centers_[i]
                distances = [np.linalg.norm(X[idx] - center) for idx in cluster_indices]
                best_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(best_idx)
                self.templates.append(images[best_idx])

                # 记录时间特征
                time_feat = self.extract_time_features(images[best_idx])
                self.template_times.append(time_feat)

        self.is_trained = True
        print(f"选择了 {len(self.templates)} 个代表性模板")

        # 显示选中的模板文件名
        print("选中的模板:")
        for idx in selected_indices[:5]:
            print(f"  - {filenames[idx]}")

        return self

    def predict(self, image_path):
        """预测：与模板比较 + 弹窗检测"""
        if not self.is_trained:
            raise ValueError("模型未训练！")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        img_resized = self.resize_image(img)
        mask = self.create_mask(img_resized.shape)

        # 1. 与模板比较
        max_similarity = 0
        best_template_idx = -1
        diff_regions = None

        for idx, template in enumerate(self.templates):
            score, diff = self.compute_masked_ssim(img_resized, template, mask)
            if score > max_similarity:
                max_similarity = score
                best_template_idx = idx
                diff_regions = diff

        # 2. 检测弹窗
        has_popup, popup_info = self.detect_popup_window(img_resized)

        # 3. 综合判断
        is_normal = (max_similarity >= self.similarity_threshold) and (not has_popup)

        # 4. 分析异常类型
        anomaly_type = []
        anomaly_details = {}

        if max_similarity < self.similarity_threshold:
            anomaly_type.append("界面异常")
            anomaly_details['similarity'] = max_similarity

            # 分析差异区域
            if diff_regions is not None:
                diff_regions = 1 - diff_regions
                # 忽略mask区域
                diff_regions[mask == 0] = 0

                # 找出主要差异区域
                threshold = np.percentile(diff_regions[diff_regions > 0], 90)
                anomaly_mask = diff_regions > threshold

                # 分析差异位置
                y_indices, x_indices = np.where(anomaly_mask)
                if len(y_indices) > 0:
                    anomaly_details['diff_center'] = (
                        int(np.mean(x_indices)),
                        int(np.mean(y_indices))
                    )

        if has_popup:
            anomaly_type.append("检测到弹窗")
            anomaly_details['popup'] = popup_info

        # 5. 返回详细结果
        result = {
            'image': os.path.basename(image_path),
            'is_normal': is_normal,
            'prediction': '正常' if is_normal else '异常',
            'similarity_score': float(max_similarity),
            'best_template': best_template_idx,
            'anomaly_types': anomaly_type,
            'anomaly_details': anomaly_details,
            'confidence': float(max_similarity) if is_normal else 1.0 - float(max_similarity)
        }

        return result

    def batch_predict(self, test_folder=None):
        """批量检测"""
        if test_folder is None:
            test_folder = "/data/temp11/程序正常"

        results = []
        anomalies = []

        print(f"\n批量检测 {test_folder} 中的图片...")

        for filename in os.listdir(test_folder):
            if filename.endswith('.jpg'):
                try:
                    img_path = os.path.join(test_folder, filename)
                    result = self.predict(img_path)
                    results.append(result)

                    if not result['is_normal']:
                        anomalies.append(result)

                except Exception as e:
                    print(f"检测 {filename} 出错: {e}")

        # 统计
        n_total = len(results)
        n_normal = sum(1 for r in results if r['is_normal'])
        n_anomaly = n_total - n_normal

        print(f"\n检测完成！")
        print(f"总数: {n_total}")
        print(f"正常: {n_normal} ({n_normal / n_total * 100:.1f}%)")
        print(f"异常: {n_anomaly} ({n_anomaly / n_total * 100:.1f}%)")

        if anomalies:
            print(f"\n发现的异常:")
            for a in anomalies[:10]:  # 最多显示10个
                print(f"  - {a['image']}: {', '.join(a['anomaly_types'])}")
                print(f"    相似度: {a['similarity_score']:.3f}")
                if 'popup' in a['anomaly_details']:
                    popup = a['anomaly_details']['popup']
                    print(f"    弹窗位置: ({popup['x']}, {popup['y']}) 大小: {popup['w']}x{popup['h']}")

        return results

    def visualize_detection(self, image_path, save_path=None):
        """可视化检测结果"""
        result = self.predict(image_path)
        img = cv2.imread(image_path)
        img_resized = self.resize_image(img)

        # 绘制检测结果
        if not result['is_normal']:
            # 绘制弹窗
            if 'popup' in result['anomaly_details']:
                popup = result['anomaly_details']['popup']
                x, y, w, h = popup['x'], popup['y'], popup['w'], popup['h']
                cv2.rectangle(img_resized, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(img_resized, "POPUP DETECTED", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # 标记异常
            cv2.putText(img_resized, f"ANOMALY: {', '.join(result['anomaly_types'])}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img_resized, "NORMAL", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示相似度
        cv2.putText(img_resized, f"Similarity: {result['similarity_score']:.3f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if save_path:
            cv2.imwrite(save_path, img_resized)
        else:
            cv2.imshow('Detection Result', img_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def save(self, filepath):
        """保存模型"""
        model_data = {
            'templates': self.templates,
            'template_times': self.template_times,
            'n_templates': self.n_templates,
            'similarity_threshold': self.similarity_threshold,
            'ignore_regions': self.ignore_regions,
            'key_regions': self.key_regions,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")

    def load(self, filepath):
        """加载模型"""
        model_data = joblib.load(filepath)
        self.templates = model_data['templates']
        self.template_times = model_data['template_times']
        self.n_templates = model_data['n_templates']
        self.similarity_threshold = model_data['similarity_threshold']
        self.ignore_regions = model_data['ignore_regions']
        self.key_regions = model_data['key_regions']
        self.is_trained = model_data['is_trained']
        print(f"模型已从 {filepath} 加载")


# 使用示例
if __name__ == "__main__":
    # 创建检测器
    detector = OneStopServiceAnomalyDetector(
        n_templates=10,  # 使用10个模板覆盖不同时间
        similarity_threshold=0.88  # 根据实际调整
    )

    # 训练
    data_folder = "/data/temp11/程序正常"
    detector.train(data_folder)

    # 保存模型
    detector.save("onestop_anomaly_detector.pkl")

    # 批量测试
    results = detector.batch_predict(data_folder)

    # 测试单张图片并可视化
    # test_image = "/data/temp11/程序正常/你的测试图片.jpg"
    # detector.visualize_detection(test_image, save_path="detection_result.jpg")
