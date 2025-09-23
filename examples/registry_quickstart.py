"""演示如何结合 VisionDataModule 与模型工厂接口。"""

import argparse
from pathlib import Path
from typing import Optional

from pyimgano import datasets, models


def main(train_dir: str, val_dir: Optional[str]) -> None:
    data_module = datasets.VisionDataModule(
        train=Path(train_dir),
        val=Path(val_dir) if val_dir is not None else None,
        loader_config=datasets.DataLoaderConfig(batch_size=8),
    )
    data_module.setup("fit")

    detector = models.create_model(
        "ae_resnet_unet",
        epoch_num=2,
        batch_size=8,
        contamination=0.05,
    )

    detector.fit(list(data_module.train_items))

    if data_module.val_items:
        scores = detector.decision_function(list(data_module.val_items))
        print(f"验证集样本数: {len(scores)}")
        print(f"分数示例: {scores[:5]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionDataModule + 模型工厂快速示例")
    parser.add_argument("train_dir", help="训练图片所在目录")
    parser.add_argument("val_dir", nargs="?", help="验证图片所在目录")
    args = parser.parse_args()
    main(args.train_dir, args.val_dir)
