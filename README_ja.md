# PyImgAno 日本語ガイド

PyImgAno は欠陥検査向けの前処理・データ拡張と、古典／深層アノマリ検知モデルをまとめたツールキットです。

## 目次

- [機能](#機能)
- [インストール](#インストール)
- [使い方](#使い方)
- [テスト](#テスト)
- [ディレクトリ構成](#ディレクトリ構成)

## 機能

- 検出器: KPCA/LOCI などの古典モデルと FastFlow/DeepSVDD/Reverse Distillation などの深層モデル。
- データ処理: MixUp/AutoAugment などの増強、欠陥検知向けフィルタ（照度補正、トップハット、Gabor）。
- ファクトリ API: `models.create_model` で統一生成、`build_augmentation_pipeline` で柔軟な前処理を構築。

## インストール

```bash
pip install -e .
# 拡散モデルを利用する場合
pip install -e .[diffusion]
```

## 使い方

```python
from pyimgano import models, utils

feature_extractor = utils.ImagePreprocessor(resize=(256, 256), output_tensor=True)
detector = models.create_model("vision_fastflow", epoch_num=5, batch_size=8)
detector.fit(["/path/to/img1.jpg", "/path/to/img2.jpg"])
```

## テスト

```bash
pip install -e .[dev]
pytest
```

## ディレクトリ構成

```text
pyimgano/
├─ models/            # 検出アルゴリズム実装
├─ utils/             # 画像処理・増強・欠陥抽出ツール
├─ datasets/          # データモジュール
├─ examples/          # サンプルコード
└─ tests/             # テストコード
```

`pyod` や `torchvision` が必須です。`diffusers` を入れると拡散モデル増強が使用できます。
