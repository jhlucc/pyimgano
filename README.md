# PyImgAno

PyImgAno 是一个面向图像异常检测的实验性工具集，现已按照 PyTorch/TF 等主流框架的结构进行模块化拆分：

- `pyimgano.datasets`: 数据集、转换与数据模块，负责图像 I/O 与 Loader 构建。
- `pyimgano.models`: 模型基类、注册表以及经典/深度算法实现，可通过工厂接口按名称创建。

## 快速上手

```python
from pyimgano import datasets, models

# 1) 构建数据模块
module = datasets.VisionDataModule(
    train="/path/to/train",
    val="/path/to/val",
    loader_config=datasets.DataLoaderConfig(batch_size=16),
)
module.setup("fit")
train_loader = module.train_dataloader()

# 2) 通过注册表实例化模型
loda = models.create_model("vision_loda", contamination=0.05, n_bins="auto")

# 3) 训练并推理
features = ...  # 预提取的特征
loda.fit(features)
```

### 数据模块 + 工厂组合示例

```python
from pyimgano import datasets, models

data = datasets.VisionDataModule(
    train="/path/to/train",
    val="/path/to/val",
    loader_config=datasets.DataLoaderConfig(batch_size=8),
)
data.setup("fit")

# 直接使用注册表构建深度模型
autoencoder = models.create_model(
    "ae_resnet_unet",
    epoch_num=5,
    batch_size=8,
    contamination=0.05,
)

# BaseVisionDeepDetector 派生类依旧接受图像路径列表
autoencoder.fit(list(data.train_items))
val_scores = autoencoder.decision_function(list(data.val_items))
```

可用模型列表：

```python
from pyimgano.models import list_models
print(list_models())  # ["ae_resnet_unet", "core_loda", "vae_conv", "vision_loda", ...]

# 也可以按标签过滤
print(list_models(tags=("deep",)))
```

欢迎根据自身场景扩展数据模块或向注册表添加新的检测器实现。
