#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: linear_regression
# Author: Mark Wang
# Date: 6/5/2016

import numpy.random as random

from regression_method import Regression


class LinearRegression(Regression):
    def __init__(self):
        Regression.__init__(self)

    def train(self, rdd_data, learn_rate=0.5, iteration=100, error=1e-8, method=None, seed=None):
        return self

    def predict(self, features):
        pass
