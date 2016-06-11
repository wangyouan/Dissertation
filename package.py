#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: package
# Author: Mark Wang
# Date: 9/5/2016

from setuptools import setup, find_packages

setup(
    name="StockInference",
    version="0.2",
    packages=find_packages(exclude=("src", 'get_data', 'StockSimulator', 'StockSimulator.RegressionMethod')),
    author="Mark Wang",
    author_email="markwang@connect.hku.hk",
    description=("A stock prediction system based on Spark"),
    keywords="Spark, Neural network",
    include_package_data=True,
    install_requires=[
        'setuptools',
        'scikit-learn',
        'numpy',
        'pandas',
        'quandl'
    ],
)
