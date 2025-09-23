from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="pyimgano",
    version="0.1.0",
    description="PyImgAno - Visual anomaly detection utilities",
    long_description=README,
    long_description_content_type="text/markdown",
    author="PyImgAno Contributors",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "torchvision",
        "Pillow",
        "opencv-python",
        "pyod",
    ],
    extras_require={
        "diffusion": [
            "diffusers",
            "transformers",
            "accelerate",
        ],
        "dev": [
            "pytest",
            "pytest-cov",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
