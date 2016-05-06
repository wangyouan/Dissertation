#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: neural_network
# Author: Mark Wang
# Date: 5/5/2016

import math

import numpy.random as random
import numpy as np

from regression_method import Regression


def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


class NeuralNetwork(Regression):
    def __init__(self, layers, seed=None, path=None):
        Regression.__init__(self)
        if path is None:
            self.logger.debug("Init weights")
            self.weights = []
            if seed is not None:
                random.seed(seed=seed)
            for i in range(1, len(layers)):
                self.weights.append(random.random_sample([layers[i - 1], layers[i]]))
        else:
            self.load_model(path)

    def train(self, rdd_data, learn_rate=0.5, iteration=100, error=1e-8, method=None):
        if method is None:
            method = self.BP

    def predict(self, features):
        temp_feature = np.array(map(sigmoid, features))
        self.logger.debug("features are %s", features)
        for weights in self.weights:
            temp_feature = temp_feature.dot(weights)

        self.logger.debug("Predict value are %s", temp_feature)
        return temp_feature


if __name__ == "__main__":
    from src import load_spark_context
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    test = NeuralNetwork([3, 4, 1], seed=1234)
    print test.predict(np.array([1.0, 2.0, 3.0]))[0]
