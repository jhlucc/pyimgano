# # 安装: pip install anomalib
# from anomalib.models import Padim, PatchCore, DFM, DFKDE
# from anomalib.data.utils import read_image
# from anomalib.deploy import TorchInferencer
# import os
# import numpy as np
#
# class AnomalibDetector:
#     """
#     使用Intel Anomalib框架 - 工业界最受欢迎的开源方案
#     集成了多种SOTA算法
#     """
#
#     def __init__(self, model_name='padim'):
#         # 支持的模型
#         self.models = {
#             'padim': Padim,
#             'patchcore': PatchCore,
#             'dfm': DFM,  # Deep Feature Modeling
#             'dfkde': DFKDE  # Deep Feature Kernel Density Estimation
#         }
#
#         self.model_name = model_name
#         self.model = None
#         self.inferencer = None
#
#     def train(self, normal_dir, val_dir=None):
#         """训练模型"""
#         from anomalib.data.folder import Folder
#         from anomalib.utils.callbacks import ModelCheckpoint
#         from pytorch_lightning import Trainer
#
#         # 准备数据
#         datamodule = Folder(
#             root=os.path.dirname(normal_dir),
#             normal_dir=os.path.basename(normal_dir),
#             abnormal_dir=None,  # 无异常数据
#             task="classification",
#             image_size=(256, 256),
#             train_batch_size=32,
#             eval_batch_size=32,
#             num_workers=0
#         )
#
#         # 创建模型
#         model_class = self.models[self.model_name]
#         self.model = model_class()
#
#         # 训练
#         trainer = Trainer(
#             max_epochs=1,  # 无监督方法通常只需要1个epoch
#             accelerator='auto',
#             devices=1,
#             callbacks=[ModelCheckpoint(mode="max", monitor="pixel_AUROC")]
#         )
#
#         trainer.fit(self.model, datamodule=datamodule)
#
#         # 创建推理器
#         self.inferencer = TorchInferencer(
#             path=trainer.checkpoint_callback.best_model_path,
#             device='cpu'
#         )
#
#         return self
#
#     def predict(self, image_path):
#         """预测单张图片"""
#         if self.inferencer is None:
#             raise ValueError("模型未训练！")
#
#         # 预测
#         result = self.inferencer.predict(image_path)
#
#         return {
#             'image': os.path.basename(image_path),
#             'is_normal': result.pred_label == 0,
#             'prediction': '正常' if result.pred_label == 0 else '异常',
#             'anomaly_score': float(result.anomaly_score),
#             'pred_mask': result.segmentation_map  # 异常区域掩码
#         }
#
#
# # 简化版本 - 不依赖完整Anomalib
# class SimplifiedAnomalibStyle:
#     """
#     Anomalib风格的实现，但不需要安装整个框架
#     """
#
#     def __init__(self, method='dfkde'):
#         self.method = method
#
#         if method == 'dfkde':
#             from sklearn.neighbors import KernelDensity
#             self.kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
#         elif method == 'dfm':
#             from sklearn.decomposition import PCA
#             self.pca = PCA(n_components=100)
#
#     def extract_deep_features(self, img_path):
#         """提取深度特征（简化版）"""
#         import torch
#         from torchvision import models, transforms
#
#         # 使用预训练模型
#         model = models.resnet18(pretrained=True)
#         model.eval()
#
#         # 移除最后的分类层
#         feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
#
#         # 预处理
#         transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#
#         # 读取图像
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_tensor = transform(img).unsqueeze(0)
#
#         # 提取特征
#         with torch.no_grad():
#             features = feature_extractor(img_tensor)
#             features = features.squeeze().numpy()
#
#         return features
#
#     def fit(self, train_folder):
#         """训练"""
#         print(f"训练 {self.method} 模型...")
#
#         features = []
#         for filename in os.listdir(train_folder):
#             if filename.endswith('.jpg'):
#                 img_path = os.path.join(train_folder, filename)
#                 feat = self.extract_deep_features(img_path)
#                 features.append(feat)
#
#         features = np.array(features)
#
#         if self.method == 'dfkde':
#             # 核密度估计
#             self.kde.fit(features)
#             # 计算训练集的分数用于阈值
#             self.train_scores = self.kde.score_samples(features)
#             self.threshold = np.percentile(self.train_scores, 5)
#         elif self.method == 'dfm':
#             # PCA降维
#             self.pca.fit(features)
#             # 计算重建误差阈值
#             reconstructed = self.pca.inverse_transform(self.pca.transform(features))
#             errors = np.mean((features - reconstructed) ** 2, axis=1)
#             self.threshold = np.percentile(errors, 95)
#
#         return self
#
#     def predict(self, img_path):
#         """预测"""
#         feat = self.extract_deep_features(img_path)
#
#         if self.method == 'dfkde':
#             score = self.kde.score_samples([feat])[0]
#             is_normal = score > self.threshold
#             anomaly_score = -score  # 转换为异常分数
#         elif self.method == 'dfm':
#             reconstructed = self.pca.inverse_transform(self.pca.transform([feat]))
#             error = np.mean((feat - reconstructed[0]) ** 2)
#             is_normal = error < self.threshold
#             anomaly_score = error
#
#         return {
#             'image': os.path.basename(img_path),
#             'is_normal': is_normal,
#             'prediction': '正常' if is_normal else '异常',
#             'anomaly_score': float(anomaly_score),
#             'method': self.method
#         }