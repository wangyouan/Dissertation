#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: linear_regression
# Author: Mark Wang
# Date: 6/5/2016

import numpy as np
import numpy.random as random

from regression_method import Regression


class LinearRegression(Regression):
    def __init__(self):
        Regression.__init__(self)

    def train(self, rdd_data, learn_rate=0.5, iteration=100, error=1e-8, method=None, seed=None):
        """
        Use for training linear regression model
        :param rdd_data: list data data type is [[feature, target], .....], features is vector liked list, current
        target only can have one output
        :param learn_rate: the learning rate of this training process
        :param iteration: the maximum iteration of this method
        :param error: target error
        :param method: method use to
        :param seed: used when self.weights is None
        :return: None
        """
        if self.weights is None:
            self.logger.debug("Init weights")
            if seed is not None:
                random.seed(seed=seed)
            self.weights = 2 * random.random_sample(len(rdd_data[0][0])) - 1

        if method is None:
            method = self.GD

        for i in range(iteration):
            self.logger.debug("Start {} iteration".format(i))
            if method == self.GD:
                if self.update_weight_using_gd(learn_rate, rdd_data, error):
                    break
            self.logger.debug("Iteration {} finished".format(i))

    def update_weight_using_gd(self, learning_rate, rdd_data, error):
        """
        Using standard gradient descent to update weights
        :param learning_rate: learning rate
        :param rdd_data: training data
        :return: None
        """
        descent = np.zeros(len(self.weights))
        loss = 0.0
        for data in rdd_data:
            loss += self.get_loss(data[0], data[1])
            temp_value = np.array(data[0], dtype=float)
            descent += 2 * (self.predict(data[0]) - data[1]) * temp_value

        loss /= len(rdd_data)
        self.logger.debug("Current loss is {:.6f}, and target error is {}".format(loss, error))
        if loss > error:
            self.weights -= learning_rate * descent
        return loss < error

    def predict(self, features):
        temp_features = np.array(features, dtype=float)
        predict_value = temp_features.dot(self.weights)
        return predict_value


if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    test = LinearRegression()
    train_data = [
        [[1, 1], 5],
        [[0, 0], 0],
        [[2, 2], 10],
        [[0.5, 0.3], 1.9],
        [[0.4, 0.5], 2.3],
        [[8, 1], 19],
        [[1, 0], 2],
        [[0, 1], 3],
        [[0.5, 1], 4],
        [[0.3, 1.1], 3.9],
        [[-1, -1], -5],
        [[-2, -2], -10],
        [[1, -2], -4],
        [[-1, 2], 4],
        [[1, -0.5], -0.5],
        [[1, -1], -1],
        [[-1.5, 1], 0],
        [[-1.1, 1], 0.8],
        [[-1.2, 1], 0.6],
        [[-4, 4], 4],
        [[2, 1], 7],
        [[0.1, 0.2], 0.8],
    ]
    test.train(train_data, learn_rate=1e-3, iteration=100, seed=1234)
    print test.predict([1, 2])
    print test.weights
