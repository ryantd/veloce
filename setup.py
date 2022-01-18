#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="phetware",
    version="0.0.1",
    author="Xiaoyu Zhai",
    author_email="zhaixiaoyu1@360.cn",
    description="phetware: An instant distributed computing ML/DL toolbox based on Ray Stack!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://prophet.qihoo.net/",
    packages=find_packages(),
    package_dir={"phetware": "phetware"}
)
