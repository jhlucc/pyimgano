"""模型注册中心，借鉴 torchvision 与 timm 的设计。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass
class ModelEntry:
    name: str
    constructor: Callable[..., Any]
    tags: tuple[str, ...]
    metadata: Dict[str, Any]


class ModelRegistry:
    """保存模型构造器的注册表。"""

    def __init__(self) -> None:
        self._registry: Dict[str, ModelEntry] = {}

    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        constructor: Callable[..., Any],
        *,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        if not overwrite and name in self._registry:
            raise KeyError(f"模型 {name!r} 已存在，请设置 overwrite=True 以覆盖。")
        entry = ModelEntry(
            name=name,
            constructor=constructor,
            tags=tuple(tags or ()),
            metadata=metadata or {},
        )
        self._registry[name] = entry

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._registry[name].constructor
        except KeyError as exc:
            available = ", ".join(sorted(self._registry)) or "<empty>"
            raise KeyError(f"未找到模型 {name!r}，当前可用: {available}") from exc

    def available(self, *, tags: Optional[Iterable[str]] = None) -> List[str]:
        if tags is None:
            return sorted(self._registry)
        tag_set = set(tags)
        return sorted(
            entry.name for entry in self._registry.values() if tag_set.issubset(entry.tags)
        )

    def info(self, name: str) -> ModelEntry:
        if name not in self._registry:
            raise KeyError(f"未找到模型 {name!r}")
        return self._registry[name]


MODEL_REGISTRY = ModelRegistry()


def register_model(
    name: str,
    *,
    tags: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """装饰器，用于在导入时自动注册模型。"""

    def decorator(constructor: Callable[..., Any]) -> Callable[..., Any]:
        MODEL_REGISTRY.register(
            name,
            constructor,
            tags=tags,
            metadata=metadata,
            overwrite=overwrite,
        )
        return constructor

    return decorator


def create_model(name: str, *args, **kwargs):
    """根据注册名称构建模型实例。"""

    constructor = MODEL_REGISTRY.get(name)
    return constructor(*args, **kwargs)


def list_models(*, tags: Optional[Iterable[str]] = None) -> List[str]:
    """列举可用模型名称。"""

    return MODEL_REGISTRY.available(tags=tags)

