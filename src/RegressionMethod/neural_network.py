#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: neural_network
# Author: Mark Wang
# Date: 5/5/2016

import math

import numpy.random as random
import numpy as np

from src import load_logger
from regression_method import Regression


def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


class NeuralNetwork(Regression):
    def __init__(self, layers, seed=None):
        Regression.__init__(self)
        self.weights = []
        if seed is not None:
            random.seed(seed=seed)
        for i in range(1, len(layers)):
            self.weights.append(random.random_sample([layers[i - 1], layers[i]]))

    def train(self, rdd_data, learn_rate=0.5, iteration=100, error=1e-8, method='back_propagation'):
        pass

    def get_loss(self, features, target, method="lse"):
        """
        Get the lose value of current target
        :param features: features of the output
        :param target: origin output
        :param method: method to measure the loss. Default method is "lse" (least square error)
        :return:
        """
        predict_value = self.predict(features)

        if method == "lse":
            return (target - predict_value) ** 2

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
