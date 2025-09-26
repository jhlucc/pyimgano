# PyImgAno

A modular computer-vision anomaly detection toolkit.
  
> Translations: [中文](README_cn.md) · [日本語](README_ja.md) · [한국어](README_ko.md)

--- 

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Testing](#testing)
- [Directory Overview](#directory-overview)

## Features

- Detectors: classical wrappers (`vision_kpca`, `vision_xgbod`, `vision_loci`, etc.) alongside deep models (`vision_fastflow`, `vision_deep_svdd`, `vision_reverse_distillation`).
- Data utilities: preprocessing helpers, augmentation registry (MixUp, CutMix, Auto/Rand/TrivialAugment, diffusion stubs), and defect-oriented filters (illumination normalization, top-hat, Gabor banks).
- Factory API: `models.create_model(name, **kwargs)` for consistent instantiation; `utils.build_augmentation_pipeline` for modular augmentation flows.

## Installation

```bash
pip install -e .
# with diffusion extras
pip install -e .[diffusion]
```

## Quick Start

```python
from pyimgano import models, utils

feature_extractor = utils.ImagePreprocessor(resize=(256, 256), output_tensor=True)
detector = models.create_model(
    "vision_fastflow",
    epoch_num=5,
    batch_size=8,
)
train_paths = ["/path/to/img1.jpg", "/path/to/img2.jpg"]
detector.fit(train_paths)
```

## Testing

```bash
pip install -e .[dev]
pytest
```

## Directory Overview

```text
pyimgano/
├─ models/            # Classical & deep detectors with registry support
├─ utils/             # Image ops, augmentations, defect preprocess, registries
├─ datasets/          # Vision data utilities (transforms, datamodule)
├─ examples/          # Usage samples (registry quickstart, ensemble demos)
└─ tests/             # Pytest suite for utils & preprocessing
```

Dependencies such as `pyod`, `torchvision`, and optional `diffusers` are declared in `setup.py`.
