# PyImgAno 한국어 안내

PyImgAno 는 결함 탐지를 위한 전처리·증강 도구와 고전/딥러닝 기반 이상 탐지 모델을 포함한 모듈형 툴킷입니다.

## 목차

- [주요 기능](#주요-기능)
- [설치](#설치)
- [빠른 사용법](#빠른-사용법)
- [테스트](#테스트)
- [디렉터리 구조](#디렉터리-구조)

## 주요 기능

- 모델: KPCA, LOCI, XGBOD 등 전통 기법과 FastFlow, DeepSVDD, Reverse Distillation 등 딥러닝 모델 제공.
- 데이터 처리: MixUp/CutMix/AutoAugment 등 증강, 조명 보정·Top-Hat·Gabor 등 결함 특화 필터.
- 공장형 API: `models.create_model`, `build_augmentation_pipeline` 으로 일관성 있는 파이프라인 구성.

## 설치

```bash
pip install -e .
# 확산 모델 기능을 포함하려면
pip install -e .[diffusion]
```

## 빠른 사용법

```python
from pyimgano import models, utils

feature_extractor = utils.ImagePreprocessor(resize=(256, 256), output_tensor=True)
detector = models.create_model("vision_fastflow", epoch_num=5, batch_size=8)
detector.fit(["/path/to/img1.jpg", "/path/to/img2.jpg"])
```

## 테스트

```bash
pip install -e .[dev]
pytest
```

## 디렉터리 구조

```text
pyimgano/
├─ models/            # 이상 탐지 모델 모음
├─ utils/             # 이미지 처리·증강·결함 전처리 유틸리티
├─ datasets/          # 데이터 모듈 및 변환
├─ examples/          # 예제 스크립트
└─ tests/             # 테스트 코드
```

필수 의존성으로 `pyod`, `torchvision` 등이 있으며, 생성 기반 증강을 위해서는 `diffusers` 를 추가로 설치하세요.
