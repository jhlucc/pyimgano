"""遍历注册表中全部模型并完成一次简易实例化。"""

from __future__ import annotations

import argparse
import pprint
from typing import Dict

import numpy as np

from pyimgano import models


class DummyFeatureExtractor:
    """最小特征提取器，用于 Vision* 模型初始化演示。"""

    def __init__(self, feature_dim: int = 32) -> None:
        self.feature_dim = feature_dim

    def extract(self, inputs):
        inputs = list(inputs)
        if not inputs:
            return np.empty((0, self.feature_dim))
        return np.random.rand(len(inputs), self.feature_dim)


# 针对需要额外参数的模型提供默认配置
MODEL_DEFAULTS: Dict[str, dict] = {
    "vision_abod": {"feature_extractor": DummyFeatureExtractor()},
    "vision_cblof": {"feature_extractor": DummyFeatureExtractor()},
    "vision_ocsvm": {"feature_extractor": DummyFeatureExtractor()},
    "vision_kpca": {"feature_extractor": DummyFeatureExtractor()},
    "core_deep_svdd": {"n_features": 64},
    "vision_deep_svdd": {"n_features": 64},
}


def instantiate_all(verbose: bool = False) -> Dict[str, str]:
    results: Dict[str, str] = {}
    for name in models.list_models():
        kwargs = MODEL_DEFAULTS.get(name, {})
        try:
            instance = models.create_model(name, **kwargs)
            cls_name = instance.__class__.__name__
            results[name] = f"OK ({cls_name})"
            if verbose:
                print(f"[✓] {name}: {cls_name}")
        except Exception as exc:  # noqa: BLE001 - 示例中收集失败信息
            results[name] = f"FAILED ({exc})"
            if verbose:
                print(f"[✗] {name}: {exc}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="对全部注册模型执行一次简易构造演示")
    parser.add_argument("--verbose", action="store_true", help="逐条输出构造结果")
    args = parser.parse_args()

    summary = instantiate_all(verbose=args.verbose)
    print("\n构造结果摘要：")
    pprint.pprint(summary)


if __name__ == "__main__":
    main()
