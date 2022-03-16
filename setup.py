#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="enscale",
    version="0.0.2",
    author="Xiaoyu Zhai",
    author_email="xiaoyu.zhai@hotmail.com",
    description="Enscale: An instant distributed computing ML/DL toolbox based on Ray Stack",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ryantd/enscale",
    packages=find_packages(),
    package_dir={"enscale": "enscale"},
)
