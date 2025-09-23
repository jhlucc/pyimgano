# PyImgAno 中文说明

PyImgAno 提供经典算法封装、深度检测器，以及适用于缺陷检测的预处理与数据增强工具。

## 目录

- [功能亮点](#功能亮点)
- [安装](#安装)
- [快速上手](#快速上手)
- [测试](#测试)
- [目录结构](#目录结构)

## 功能亮点

- 检测器：包含 KPCA、XGBOD、LOCI 等传统算法，以及 FastFlow、DeepSVDD、Reverse Distillation 等深度模型。
- 数据工具：图像预处理、增强注册表（MixUp/CutMix/AutoAugment 等）以及缺陷检测专用滤波（光照校正、Top-Hat、Gabor）。
- 工厂接口：`models.create_model` 统一管理模型；`utils.build_augmentation_pipeline` 快速构建增强流程。

## 安装

```bash
pip install -e .
# 或安装扩展的扩散模型支持
pip install -e .[diffusion]
```

## 快速上手

```python
from pyimgano import models, utils

feature_extractor = utils.ImagePreprocessor(resize=(256, 256), output_tensor=True)
detector = models.create_model("vision_fastflow", epoch_num=5, batch_size=8)
train_paths = ["/path/to/img1.jpg", "/path/to/img2.jpg"]
detector.fit(train_paths)
```

## 测试

```bash
pip install -e .[dev]
pytest
```

## 目录结构

```text
pyimgano/
├─ models/            # 视觉异常检测模型集合
├─ utils/             # 图像处理、增强、缺陷预处理等工具
├─ datasets/          # 数据加载与转换模块
├─ examples/          # 示例脚本与用法演示
└─ tests/             # 单元测试
```

项目依赖 `pyod`、`torchvision` 等库；如需生成式增强，请额外安装 `diffusers`。
