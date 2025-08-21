# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pyod.models.deep_learning import BaseDeepLearningDetector


class ImageDataset(Dataset):
    """一个简单的图像数据集类，用于从路径加载图片。"""

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # 以 RGB 格式打开图片，避免单通道灰度图或带Alpha通道的图片出错
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            # 返回图像本身作为数据和目标，这在自监督学习（如自编码器）中很常见
            return img, img
        except Exception as e:
            print(f"加载图片时出错 {self.image_paths[idx]}: {e}")
            # 返回一个占位符张量
            return torch.zeros((3, 224, 224)), torch.zeros((3, 224, 224))


# --- 核心基类 ---
class BaseVisionDeepDetector(BaseDeepLearningDetector):
    """
    所有基于深度学习的端到端视觉异常检测算法的基类。
    本类继承自 PyOD 的 BaseDeepLearningDetector，复用了其完整的训练框架，
    """

    def __init__(self, contamination=0.1, preprocessing=True,
                 lr=1e-3, epoch_num=10, batch_size=32,
                 optimizer_name='adam', criterion_name='mse',
                 device=None, random_state=42, verbose=1,
                 train_transform=None, eval_transform=None, **kwargs):

        # 调用父类 (pyod.BaseDeepLearningDetector) 的构造函数
        super(BaseVisionDeepDetector, self).__init__(
            contamination=contamination,
            preprocessing=preprocessing,
            lr=lr,
            epoch_num=epoch_num,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            criterion_name=criterion_name,
            device=device,
            random_state=random_state,
            verbose=verbose,
            **kwargs
        )

        if train_transform is not None:
            self.train_transform = train_transform
        else:
            if self.preprocessing:
                # 默认训练时可以加入轻微的数据增强
                self.train_transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])
            else:
                self.train_transform = transforms.ToTensor()
        # 为评估过程设置 transform
        if eval_transform is not None:
            self.eval_transform = eval_transform
        else:
            if self.preprocessing:
                # 评估时不应有数据增强，保证结果一致性
                self.eval_transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])
            else:
                self.eval_transform = transforms.ToTensor()

    def fit(self, X, y=None):
        """
        【特色功能 3: 重写 fit 方法以处理图像路径】
        使用正常的、无缺陷的图像数据来拟合检测器。

        Parameters
        ----------
        X : list of str
            输入的训练样本，必须是图像文件路径的列表。
        """
        # 1. 构建模型
        self.model = self.build_model()
        # 2. 创建图像数据加载器 (使用训练专用的 transform)
        train_dataset = ImageDataset(image_paths=X, transform=self.train_transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # 3. 准备训练 (来自父类的方法)
        self.training_prepare()

        # 4. 执行训练循环 (来自父类的方法，它会调用我们子类实现的 training_forward)
        if self.verbose: print(f"开始在 {self.device} 设备上进行训练...")
        self.train(train_loader)
        if self.verbose: print("训练完成。")

        # 5. 计算训练集上的异常分数
        if self.verbose: print("正在计算训练集上的异常分数...")
        self.decision_scores_ = self.decision_function(X)

        # 6. 调用 PyOD 的方法来计算阈值和标签 (继承来的免费功能)
        self._process_decision_scores()
        return self

    def decision_function(self, X, batch_size=None):

        current_batch_size = batch_size if batch_size is not None else self.batch_size

        # 创建图像数据加载器
        eval_dataset = ImageDataset(image_paths=X, transform=self.eval_transform)
        eval_loader = DataLoader(eval_dataset, batch_size=current_batch_size, shuffle=False)

        # 调用父类的评估方法 evaluating_forward
        scores = self.evaluate(eval_loader)
        return scores

