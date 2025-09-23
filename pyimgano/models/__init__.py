"""模型模块，提供统一的工厂与注册接口。"""

from importlib import import_module
from typing import Iterable
import warnings

from .baseml import BaseVisionDetector
from .baseCv import BaseVisionDeepDetector
from .registry import MODEL_REGISTRY, create_model, list_models, register_model


def _auto_import(modules: Iterable[str]) -> None:
    """按需导入并触发注册表装饰器。"""

    for module_name in modules:
        try:
            import_module(f"{__name__}.{module_name}")
        except Exception as exc:  # noqa: BLE001 - 记录导入失败信息
            warnings.warn(
                f"加载模型模块 {module_name!r} 失败: {exc}",
                RuntimeWarning,
            )


_auto_import(
    [
        "abod",
        "ae",
        "ae1svm",
        "alad",
        "cblof",
        "dbscan",
        "efficientad",
        "Isolationforest",
        "kpca",
        "k_means",
        "lof",
        "loda",
        "ocsvm",
        "one_svm_cnn",
        "padim",
        "ssim",
        "ssim_struct",
        "vae",
    ]
)

from .ae import OptimizedAEDetector  # noqa: E402  # re-export常用模型
from .loda import VisionLODA  # noqa: E402
from .vae import VAEAnomalyDetector  # noqa: E402

__all__ = [
    "BaseVisionDetector",
    "BaseVisionDeepDetector",
    "MODEL_REGISTRY",
    "create_model",
    "list_models",
    "register_model",
    "VisionLODA",
    "OptimizedAEDetector",
    "VAEAnomalyDetector",
]
