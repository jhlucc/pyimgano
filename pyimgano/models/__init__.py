"""模型模块，统一暴露基础检测器接口。"""

from .baseml import BaseVisionDetector
from .baseCv import BaseVisionDeepDetector

__all__ = [
    "BaseVisionDetector",
    "BaseVisionDeepDetector",
]
