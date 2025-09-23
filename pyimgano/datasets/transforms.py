"""常用图像增广与预处理流水线，参考 torchvision 的组织方式。"""

from __future__ import annotations

from typing import Iterable, Tuple

from torchvision import transforms


__all__ = [
    "default_train_transforms",
    "default_eval_transforms",
    "to_tensor_normalized",
]


def to_tensor_normalized(
    mean: Iterable[float] = (0.485, 0.456, 0.406),
    std: Iterable[float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """构建标准张量化并归一化的转换。"""

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def default_train_transforms(
    resize: Tuple[int, int] = (256, 256),
    crop_size: int = 224,
    horizontal_flip: bool = True,
) -> transforms.Compose:
    """默认训练阶段的数据增强策略。"""

    augmentations = [
        transforms.Resize(resize),
        transforms.RandomCrop(crop_size),
    ]
    if horizontal_flip:
        augmentations.append(transforms.RandomHorizontalFlip())
    augmentations.append(to_tensor_normalized())
    return transforms.Compose(augmentations)


def default_eval_transforms(
    resize: Tuple[int, int] = (256, 256),
    crop_size: int = 224,
) -> transforms.Compose:
    """默认验证/测试阶段的预处理流程。"""

    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop_size),
        to_tensor_normalized(),
    ])

