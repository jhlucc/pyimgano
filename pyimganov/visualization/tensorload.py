import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from one_svm import ImageAnomalyDetector

class AnomalyDetectorEvaluator:
    def __init__(self, detector):
        self.detector = detector

    def visualize_anomaly_scores(self, test_folder, true_labels=None):
        """可视化异常分数分布"""
        results = self.detector.batch_predict(test_folder)
        scores = [r['anomaly_score'] for r in results]

        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='Decision Boundary')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Anomaly Scores')
        plt.legend()
        plt.show()

    def evaluate_with_labels(self, test_folder, labels_dict):
        """如果有标签，评估性能"""
        results = self.detector.batch_predict(test_folder)

        y_true = []
        y_pred = []

        for result in results:
            filename = result['filename']
            if filename in labels_dict:
                y_true.append(labels_dict[filename])  # 0: normal, 1: anomaly
                y_pred.append(0 if result['is_normal'] else 1)

        # 计算指标
        print("\n分类报告:")
        print(classification_report(y_true, y_pred,
                                    target_names=['Normal', 'Anomaly']))

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()


# 快速开始脚本
def quick_start():
    # 对于你的数据
    detector = ImageAnomalyDetector(feature_type='combined', nu=0.05)
    detector.train("/Computer/data/temp11/程序正常")
    detector.save_model("my_anomaly_detector.pkl")

    # 测试
    result = detector.predict("/Computer/data/temp11/程序正常/0a6c84ef7e1942529f46fea50ed5dfab.jpg")
    print(f"检测结果: {result['prediction']}")
    print(f"异常分数: {result['anomaly_score']:.4f}")


# 运行
quick_start()