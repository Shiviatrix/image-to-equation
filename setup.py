from setuptools import setup, find_packages

setup(
    name="image_to_equation",
    version="0.1.0",
    description="A Neural-Symbolic differentiable image codec.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.5.0",
    ],
    entry_points={
        "console_scripts": [
            "i2eenc=scripts.i2eenc:main",
            "i2edec=scripts.i2edec:main",
        ],
    },
)
