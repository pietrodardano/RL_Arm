"""Setup script for RL_Arm package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rl_arm",
    version="0.1.0",
    author="Pietro Dardano",
    description="Reinforcement Learning for Robot Arm Manipulators using IsaacLab",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pietrodardano/RL_Arm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23.0",
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "black>=22.10.0",
            "isort>=5.11.0",
            "flake8>=6.0.0",
            "mypy>=0.991",
        ],
    },
)
