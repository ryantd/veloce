#!/usr/bin/env python

import os
import re
from setuptools import setup, find_packages

ROOT_DIR = os.path.dirname(__file__)

def find_version(*filepath):
    # Extract version information from filepath
    with open(os.path.join(ROOT_DIR, *filepath)) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="veloce",
    version=find_version("veloce", "version.py"),
    author="Xiaoyu Zhai",
    author_email="xiaoyu.zhai@hotmail.com",
    description="Veloce: An instant distributed computing library based on Ray stack",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ryantd/veloce",
    packages=find_packages(),
    package_dir={
        "veloce": "veloce",
        "docs": "docs",
    },
    install_requires=[
        "torch>=1.9.1",
        "ray>=1.9.2,<=1.10",
        "pyarrow>=6.0.1",
        "pandas>=1.3.5,<=1.4.1",
        "requests>=2.23.0",
    ],
    python_requires=">=3.7.1",
)
