"""图像数据集抽象，集中管理数据读取逻辑。"""

from __future__ import annotations

import os
from typing import Callable, Iterable, Optional, Sequence, Tuple

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset


class VisionImageDataset(Dataset):
    """视觉检测任务常用数据集。

    - 默认以 PIL 方式加载图像，返回 `(image, image)`，适用于自编码器等重建任务。
    - 任何加载错误都会返回指定形状的零张量以保证训练流程不断。"""

    def __init__(
        self,
        image_paths: Sequence[str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        fallback_shape: Tuple[int, int, int] = (3, 224, 224),
    ) -> None:
        self.image_paths = list(image_paths)
        self.transform = transform
        self.target_transform = target_transform
        self.fallback_shape = fallback_shape

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]

        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            target = image
            if self.target_transform is not None:
                target = self.target_transform(target)
            return image, target
        except Exception as exc:  # noqa: BLE001 - 需打印完整错误信息便于排查
            print(f"加载图片时出错 {image_path}: {exc}")
            fallback = torch.zeros(self.fallback_shape, dtype=torch.float32)
            return fallback, fallback.clone()


class ImagePathDataset(Dataset):
    """读取图像并回传文件信息的通用数据集。"""

    def __init__(
        self,
        image_paths: Iterable[str],
        transform: Optional[Callable] = None,
        return_full_path: bool = False,
    ) -> None:
        self.image_paths = list(image_paths)
        self.transform = transform
        self.return_full_path = return_full_path

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        label = image_path if self.return_full_path else os.path.basename(image_path)
        return image, label
