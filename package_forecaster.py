#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: package_forecaster
# Author: Mark Wang
# Date: 18/10/2016

from setuptools import setup, find_packages

setup(
    name="stockforecaster",
    version="0.1",
    packages=find_packages(exclude=("src", 'get_data', 'StockSimulator', 'StockSimulator.RegressionMethod',
                                    'StockInference')),
    author="Mark Wang",
    author_email="markwang@connect.hku.hk",
    description="A stock prediction system based on Spark",
    keywords="Spark, Neural network, Stock Forecaster",
    install_requires=[
        'setuptools',
        'scikit-learn',
        'numpy',
        'pandas',
        'quandl',
        'keras',
        'elephas'
    ],
    classifiers=[
        "Programming Language :: Python :: 2.7"
    ],
)
