#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name="pinn",
    version="0.1",
    description="Power in Network Neuroscience",
    author="Koen Helwegen",
    author_email="kg.helwegen@gmail.com",
    packages=find_packages(),
    install_requires=[
        "statsmodels",
    ],
    extras_require={
        "test": [
            "black",
            "flake8",
            "isort",
            "pytest",
            "pytype",
        ],
    },
)
