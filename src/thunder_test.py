#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: thunder_test
# Author: Mark Wang
# Date: 16/6/2016

import numpy as np
from regression import CustomRegression
from sklearn.ensemble import RandomForestRegressor
from thunder import series

from pyspark import SparkContext
sc = SparkContext()

# generate data
from sklearn.datasets import make_regression
X = series.fromrandom((100, 3), engine=sc)
Y = series.fromrandom(100, engine=sc)

algorithm = CustomRegression(RandomForestRegressor())

# X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
# y = np.array([1, 1, 0, 0])



model = algorithm.fit(X, Y)

result = model.predict(X[:10])

print
