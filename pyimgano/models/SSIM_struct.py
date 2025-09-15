import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import joblib
from tqdm import tqdm
import json
from collections import defaultdict


class MultiTemplatePopupStructDetector:
    """
    模板规则弹窗检测器
    - 支持多个正常界面模板
    - 基于结构特征而非颜色
    - 重点检测弹窗、错误提示等异常
    """

    def __init__(self, structure_threshold=0.75, popup_min_area=3000, resize_shape=(640, 480)):
        self.structure_threshold = structure_threshold
        self.popup_min_area = popup_min_area
        self.resize_shape = resize_shape
        self.templates = {}
        self.template_structures = {}
        self.is_trained = False

    def train_from_organized_data(self, data_folder, samples_per_template=20):
        """从已组织好的数据训练"""
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"数据目录不存在: {data_folder}")

        print(f"开始从组织好的数据训练...")
        print(f"数据目录: {data_folder}")
        print(f"每个模板采样数: {samples_per_template}")

        # 检测数据组织方式
        subdirs = [d for d in os.listdir(data_folder)
                   if os.path.isdir(os.path.join(data_folder, d))]

        if subdirs:
            print(f"检测到 {len(subdirs)} 个模板文件夹")
            self._train_from_folders(data_folder, subdirs, samples_per_template)
        else:
            print("使用文件名前缀方式组织")
            self._train_from_prefixes(data_folder, samples_per_template)

        self.is_trained = True
        print(f"\n训练完成！共加载 {len(self.templates)} 个模板")
        return self

    def _train_from_folders(self, data_folder, template_folders, samples_per_template):
        """从文件夹结构训练"""
        for template_name in tqdm(template_folders, desc="处理模板"):
            template_path = os.path.join(data_folder, template_name)

            # 获取该模板的所有图片
            jpg_files = [f for f in os.listdir(template_path) if f.endswith('.jpg')]

            if not jpg_files:
                print(f"警告：模板 {template_name} 没有找到图片")
                continue

            print(f"\n处理模板 {template_name}: {len(jpg_files)} 张图片")

            # 选择样本
            selected_files = self._select_samples(jpg_files, samples_per_template)

            # 提取该模板的代表性特征 - 修正：传入正确的目录路径
            template_features = self._extract_template_features(
                template_path, selected_files, template_name
            )

            if template_features:
                self.templates[template_name] = template_features['representative_img']
                self.template_structures[template_name] = template_features['features']

    def _train_from_prefixes(self, data_folder, samples_per_template):
        """从文件名前缀训练"""
        all_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]

        # 按前缀分组
        template_files = defaultdict(list)
        for filename in all_files:
            # 假设模板名是下划线前的部分
            prefix = filename.split('_')[0]
            template_files[prefix].append(filename)

        print(f"检测到 {len(template_files)} 个不同的模板前缀")

        for template_name, files in tqdm(template_files.items(), desc="处理模板"):
            print(f"\n处理模板 {template_name}: {len(files)} 张图片")

            # 选择样本
            selected_files = self._select_samples(files, samples_per_template)

            # 提取该模板的代表性特征
            template_features = self._extract_template_features(
                data_folder, selected_files, template_name
            )

            if template_features:
                self.templates[template_name] = template_features['representative_img']
                self.template_structures[template_name] = template_features['features']

    def _select_samples(self, files, samples_per_template):
        """智能选择样本"""
        if len(files) <= samples_per_template:
            return files

        # 如果文件很多，均匀采样
        step = len(files) // samples_per_template
        selected = []
        for i in range(0, len(files), step):
            if len(selected) < samples_per_template:
                selected.append(files[i])

        return selected

    def _extract_template_features(self, base_path, files, template_name):
        """提取模板的代表性特征"""
        all_features = []
        all_images = []

        if not os.path.isdir(base_path):
            raise ValueError(f"base_path必须是目录: {base_path}")

        # 读取所有样本
        for filename in files:
            img_path = os.path.join(base_path, filename)

            if not os.path.exists(img_path):
                print(f"警告：文件不存在 {img_path}")
                continue

            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, self.resize_shape)
                features = self.extract_structure_features(img_resized)
                all_features.append(features)
                all_images.append(img_resized)

        if not all_features:
            return None

        print(f"  成功读取 {len(all_features)} 张图片")

        #  选择最具代表性的图片（与其他图片相似度最高的）
        if len(all_images) > 1:
            similarity_scores = []

            for i in range(len(all_features)):
                total_similarity = 0
                for j in range(len(all_features)):
                    if i != j:
                        sim = self.compute_structure_similarity(
                            all_features[i], all_features[j]
                        )
                        total_similarity += sim

                avg_similarity = total_similarity / (len(all_features) - 1)
                similarity_scores.append(avg_similarity)

            # 选择平均相似度最高的
            best_idx = np.argmax(similarity_scores)
            representative_img = all_images[best_idx]
            representative_features = all_features[best_idx]

            print(f"  选择第 {best_idx + 1} 张图片作为代表（平均相似度: {similarity_scores[best_idx]:.3f}）")
        else:
            representative_img = all_images[0]
            representative_features = all_features[0]

        return {
            'representative_img': representative_img,
            'features': representative_features,
            'num_samples': len(all_images)
        }

    def extract_structure_features(self, img):
        """提取结构特征（忽略颜色）"""
        # 输入验证
        if img is None or img.size == 0:
            raise ValueError("输入图像为空")

        # 1. 转换为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. 提取多种结构特征
        features = {}

        # 边缘特征
        edges = cv2.Canny(gray, 50, 150)
        features['edges'] = edges

        # 梯度特征
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        features['gradient'] = gradient_magnitude

        # 形态学特征（提取主要结构）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morphology = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        features['morphology'] = morphology

        # 二值化特征（提取文字和UI元素）
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        features['binary'] = binary

        return features

    def compute_structure_similarity(self, features1, features2):
        """计算结构相似度（多特征融合）"""
        try:
            similarities = []

            # 边缘相似度
            edge_sim = ssim(features1['edges'], features2['edges'], data_range=255)
            similarities.append(edge_sim * 0.4)  # 权重0.4

            # 梯度相似度
            grad1_norm = features1['gradient'] / (np.max(features1['gradient']) + 1e-6)
            grad2_norm = features2['gradient'] / (np.max(features2['gradient']) + 1e-6)
            grad_sim = ssim(grad1_norm, grad2_norm, data_range=1.0)
            similarities.append(grad_sim * 0.3)  # 权重0.3

            # 形态学相似度
            morph_sim = ssim(features1['morphology'], features2['morphology'], data_range=255)
            similarities.append(morph_sim * 0.2)  # 权重0.2

            # 二值特征相似度
            binary_sim = ssim(features1['binary'], features2['binary'], data_range=255)
            similarities.append(binary_sim * 0.1)  # 权重0.1

            # 加权平均
            total_similarity = sum(similarities)

            return total_similarity

        except Exception as e:
            print(f"计算相似度时出错: {e}")
            return 0.0

    def detect_popup_windows(self, img, bg_features=None):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        edges = cv2.Canny(gray, 100, 200)

        if bg_features is not None:
            # 仅用“与背景差分”
            current_edges = cv2.Canny(gray, 50, 150)
            diff = cv2.absdiff(current_edges, bg_features['edges'])
            _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            mask = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel, iterations=1)
        else:
            # 无背景时再退化到形态学闭运算
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area_min = 0.05 * h * w  # 5%
        area_max = 0.40 * h * w  # 40%

        cand = []
        for c in contours:
            x, y, ww, hh = cv2.boundingRect(c)
            area = ww * hh
            if not (area_min <= area <= area_max):
                continue

            cx, cy = x + ww / 2, y + hh / 2
            # 更严格的中心区域（25%~75%）
            if not (0.25 * w < cx < 0.75 * w and 0.25 * h < cy < 0.75 * h):
                continue

            rect = cv2.minAreaRect(c)
            ra = max(rect[1][0] * rect[1][1], 1)
            rectangularity = area / ra

            roi = edges[y:y + hh, x:x + ww]
            # 只看“边框”密度（外1/10 边缘环带），更接近有边框的弹窗
            bw = max(1, ww // 10);
            bh = max(1, hh // 10)
            border = np.zeros_like(roi, dtype=np.uint8)
            border[:bh, :] = roi[:bh, :];
            border[-bh:, :] = roi[-bh:, :]
            border[:, :bw] = roi[:, :bw];
            border[:, -bw:] = roi[:, -bw:]
            border_density = np.mean(border > 0)

            score = 0
            if rectangularity > 0.92: score += 35
            if 0.6 < ww / hh < 1.8:     score += 25
            if border_density > 0.03: score += 25
            # 可选：与背景差分区域占比（只有有 bg_features 时才算分）
            if bg_features is not None:
                roi_diff = mask[y:y + hh, x:x + ww]
                if np.mean(roi_diff > 0) > 0.15:
                    score += 15

            if score >= 70:
                cand.append({'x': x, 'y': y, 'w': ww, 'h': hh, 'area': area, 'score': score})

        # 用 IoU 做 NMS
        def iou(a, b):
            x1 = max(a['x'], b['x']);
            y1 = max(a['y'], b['y'])
            x2 = min(a['x'] + a['w'], b['x'] + b['w'])
            y2 = min(a['y'] + a['h'], b['y'] + b['h'])
            if x2 <= x1 or y2 <= y1: return 0.0
            inter = (x2 - x1) * (y2 - y1)
            union = a['area'] + b['area'] - inter
            return inter / union

        cand.sort(key=lambda p: p['score'], reverse=True)
        kept = []
        for p in cand:
            if all(iou(p, q) < 0.5 for q in kept):
                kept.append(p)

        return len(kept) > 0, kept

    def predict(self, image_path):
        """预测单张图片"""
        if not self.is_trained:
            raise ValueError("模型未训练！")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 统一尺寸
        img_resized = cv2.resize(img, self.resize_shape)

        # 提取结构特征
        current_features = self.extract_structure_features(img_resized)

        # 与所有模板比较，找最相似的
        max_similarity = 0
        best_template_name = None
        best_template_features = None

        # 遍历字典格式的模板
        for template_name, template_features in self.template_structures.items():
            similarity = self.compute_structure_similarity(current_features, template_features)
            if similarity > max_similarity:
                max_similarity = similarity
                best_template_name = template_name
                best_template_features = template_features

        #  检测弹窗
        has_popup, popups = self.detect_popup_windows(img_resized, best_template_features)

        # 判断是否正常
        structure_normal = max_similarity >= self.structure_threshold
        is_normal = structure_normal and not has_popup

        #  构建结果
        result = {
            'image': os.path.basename(image_path),
            'is_normal': is_normal,
            'prediction': '正常' if is_normal else '异常',
            'best_template': best_template_name,
            'structure_similarity': float(max_similarity),
            'has_popup': has_popup,
            'popup_count': len(popups),
            'anomaly_types': []
        }

        # 分析异常类型
        if not structure_normal:
            result['anomaly_types'].append('界面结构异常')
        if has_popup:
            result['anomaly_types'].append(f'检测到{len(popups)}个弹窗')
            result['popups'] = popups

        return result

    def batch_predict(self, test_folder):
        """批量检测"""
        if not os.path.exists(test_folder):
            raise FileNotFoundError(f"测试目录不存在: {test_folder}")

        results = []
        anomalies = []

        print(f"\n批量检测 {test_folder} 中的图片...")

        jpg_files = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]

        for filename in tqdm(jpg_files, desc="检测进度"):
            try:
                img_path = os.path.join(test_folder, filename)
                result = self.predict(img_path)
                results.append(result)

                if not result['is_normal']:
                    anomalies.append(result)

            except Exception as e:
                print(f"\n检测 {filename} 出错: {e}")
                import traceback
                traceback.print_exc()

        # 统计
        n_total = len(results)
        n_normal = sum(1 for r in results if r['is_normal'])
        n_anomaly = n_total - n_normal

        print(f"\n=== 检测结果统计 ===")
        print(f"总数: {n_total}")
        print(f"正常: {n_normal} ({n_normal / n_total * 100:.1f}%)")
        print(f"异常: {n_anomaly} ({n_anomaly / n_total * 100:.1f}%)")

        # 模板使用统计
        template_usage = {}
        for r in results:
            template = r.get('best_template', 'Unknown')
            template_usage[template] = template_usage.get(template, 0) + 1

        print(f"\n=== 模板匹配统计 ===")
        for template, count in sorted(template_usage.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"模板 {template}: 匹配 {count} 次 ({count / n_total * 100:.1f}%)")

        if anomalies:
            print(f"\n=== 检测到的异常（前10个）===")
            for a in sorted(anomalies, key=lambda x: len(x.get('popups', [])), reverse=True)[:10]:
                print(f"\n{a['image']}:")
                print(f"  异常类型: {', '.join(a['anomaly_types'])}")
                print(f"  结构相似度: {a['structure_similarity']:.3f}")
                print(f"  最佳匹配模板: {a['best_template']}")
                if a['has_popup']:
                    print(f"  弹窗数量: {a['popup_count']}")
                    for i, popup in enumerate(a['popups'][:3]):  # 最多显示3个
                        print(f"    弹窗{i + 1}: 位置({popup['x']}, {popup['y']}), "
                              f"大小({popup['w']}x{popup['h']}), 分数:{popup['score']}")

        return results

    def save_template_info(self, info_file="template_info.json"):
        """保存模板信息到JSON文件"""
        template_info = {
            'num_templates': len(self.templates),
            'templates': list(self.templates.keys()),
            'structure_threshold': self.structure_threshold,
            'popup_min_area': self.popup_min_area,
            'resize_shape': self.resize_shape
        }

        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(template_info, f, ensure_ascii=False, indent=2)

        print(f"模板信息已保存到 {info_file}")

    def save(self, filepath):
        """保存模型（不保存完整图像，只保存特征）"""
        # 只保存特征
        templates_features_only = {}
        for name, img in self.templates.items():
            # 重新提取特征而不是保存图像
            templates_features_only[name] = self.extract_structure_features(img)

        model_data = {
            'template_structures': self.template_structures,  # 这已经是特征
            'structure_threshold': self.structure_threshold,
            'popup_min_area': self.popup_min_area,
            'resize_shape': self.resize_shape,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")

    def load(self, filepath):
        """加载模型"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        model_data = joblib.load(filepath)

        # 兼容旧版本
        if 'templates' in model_data:
            self.templates = model_data['templates']
        else:
            self.templates = {}  # 新版本不保存图像

        self.template_structures = model_data['template_structures']
        self.structure_threshold = model_data['structure_threshold']
        self.popup_min_area = model_data['popup_min_area']
        self.resize_shape = model_data.get('resize_shape', (640, 480))
        self.is_trained = model_data['is_trained']
        print(f"模型已从 {filepath} 加载")
        print(f"加载了 {len(self.template_structures)} 个模板")


# 使用示例
if __name__ == "__main__":
    # 创建检测器
    detector = MultiTemplatePopupStructDetector(
        structure_threshold=0.6,
        popup_min_area=3000
    )

    # 训练
    try:
        organized_data = "/data/temp11/正常"
        detector.train_from_organized_data(organized_data, samples_per_template=20)

        # 保存模型和模板信息
        detector.save("multi_template_popup_detector.pkl")
        detector.save_template_info("template_info.json")

    except Exception as e:
        print(f"训练出错: {e}")
        import traceback

        traceback.print_exc()

    # 检测单张图片
    try:
        test_image = "/data/temp11/正常/1/0bbe53cea48546498e414cd070adc2e1.jpg"
        result = detector.predict(test_image)
        print(f"检测结果: {result['prediction']}")
        print(f"最佳匹配模板: {result['best_template']}")
    except Exception as e:
        print(f"检测出错: {e}")