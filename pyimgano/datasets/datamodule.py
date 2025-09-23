"""数据模块工具，提供类似 PyTorch Lightning 的数据装载体验。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from .image import ImagePathDataset, VisionImageDataset
from .transforms import default_eval_transforms, default_train_transforms

__all__ = [
    "VisionDataModule",
    "DataLoaderConfig",
]


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class DataLoaderConfig:
    """控制 DataLoader 行为的配置封装。"""

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: Optional[bool] = None
    drop_last: bool = False
    shuffle: bool = True

    def resolve(self, device: Optional[torch.device] = None) -> dict:
        pin_memory = self.pin_memory
        if pin_memory is None and device is not None:
            pin_memory = device.type == "cuda"
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": pin_memory,
            "drop_last": self.drop_last,
            "shuffle": self.shuffle,
        }


class VisionDataModule:
    """统一管理视觉任务数据加载的模块。

    - 支持传入路径列表或目录，自动筛选常见图像扩展名。
    - 提供默认的训练/验证转换，与 BaseVisionDeepDetector 保持对齐。
    - 可选地返回 `(image, image)` 形式的数据，用于重建类任务。
    """

    def __init__(
        self,
        train: Optional[Iterable[str]] = None,
        val: Optional[Iterable[str]] = None,
        test: Optional[Iterable[str]] = None,
        *,
        reconstruction: bool = True,
        train_transform=None,
        eval_transform=None,
        device: Optional[torch.device] = None,
        loader_config: Optional[DataLoaderConfig] = None,
    ) -> None:
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reconstruction = reconstruction
        self.train_transform = train_transform or default_train_transforms()
        self.eval_transform = eval_transform or default_eval_transforms()
        self.loader_config = loader_config or DataLoaderConfig()

        self.train_paths = self._normalize_source(train)
        self.val_paths = self._normalize_source(val)
        self.test_paths = self._normalize_source(test)

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        """根据阶段初始化对应数据集。"""

        if stage in (None, "fit") and self.train_paths is not None:
            self._train_dataset = self._build_dataset(self.train_paths, train=True)
            if self.val_paths:
                self._val_dataset = self._build_dataset(self.val_paths, train=False)
        if stage in (None, "test") and self.test_paths is not None:
            self._test_dataset = self._build_dataset(self.test_paths, train=False)

    def train_dataloader(self) -> DataLoader:
        self._ensure_dataset(self._train_dataset, "train")
        cfg = self.loader_config.resolve(self._device)
        cfg["shuffle"] = True
        return DataLoader(self._train_dataset, **cfg)

    def val_dataloader(self) -> DataLoader:
        self._ensure_dataset(self._val_dataset, "validation")
        cfg = self.loader_config.resolve(self._device)
        cfg["shuffle"] = False
        return DataLoader(self._val_dataset, **cfg)

    def test_dataloader(self) -> DataLoader:
        self._ensure_dataset(self._test_dataset, "test")
        cfg = self.loader_config.resolve(self._device)
        cfg["shuffle"] = False
        return DataLoader(self._test_dataset, **cfg)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_dataset(self, paths: Sequence[str], *, train: bool) -> torch.utils.data.Dataset:
        dataset_cls = VisionImageDataset if self.reconstruction else ImagePathDataset
        transform = self.train_transform if train else self.eval_transform
        return dataset_cls(paths, transform=transform)

    def _normalize_source(self, source: Optional[Iterable[str]]) -> Optional[Sequence[str]]:
        if source is None:
            return None
        if isinstance(source, (str, os.PathLike)):
            return self._scan_directory(Path(source))
        return tuple(source)

    def _scan_directory(self, directory: Path) -> Sequence[str]:
        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")
        paths = [
            str(path)
            for path in sorted(directory.iterdir())
            if path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not paths:
            raise ValueError(f"在 {directory} 中找不到支持的图像文件 {SUPPORTED_EXTENSIONS}")
        return paths

    @staticmethod
    def _ensure_dataset(dataset, stage: str) -> None:
        if dataset is None:
            raise RuntimeError(f"请先在 setup 阶段准备 {stage} 数据集")

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def train_dataset(self):
        self._ensure_dataset(self._train_dataset, "train")
        return self._train_dataset

    @property
    def val_dataset(self):
        self._ensure_dataset(self._val_dataset, "validation")
        return self._val_dataset

    @property
    def test_dataset(self):
        self._ensure_dataset(self._test_dataset, "test")
        return self._test_dataset

    @property
    def train_items(self) -> Sequence[str]:
        return self.train_paths

    @property
    def val_items(self) -> Optional[Sequence[str]]:
        return self.val_paths

    @property
    def test_items(self) -> Optional[Sequence[str]]:
        return self.test_paths
