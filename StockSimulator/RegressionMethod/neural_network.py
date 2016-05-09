#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: neural_network
# Author: Mark Wang
# Date: 5/5/2016

import numpy.random as random
import numpy as np

from StockSimulator.RegressionMethod.regression_method import Regression


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_dot(x):
    """ return the derivative of sigmoid function """
    x = sigmoid(x)
    return x * (1 - x)


def de_sigmoid(x):
    return np.log(x / (1 - x))


def tanh(x):
    return np.tanh(x)


def tanh_dot(x):
    return 1.0 - x ** 2


class NeuralNetwork(Regression):
    def __init__(self, layers, seed=None, path=None, activation_func=None, activation_func_dot=None, bias=0.0):
        Regression.__init__(self)
        if activation_func is None:
            self.af = sigmoid
            self.afd = sigmoid_dot
        else:
            self.af = activation_func
            self.afd = activation_func_dot

        self.bias = bias
        if path is None:
            self.logger.debug("Init weights")
            self.weights = []
            if seed is not None:
                random.seed(seed=seed)

            # Add one bias unit to the output
            i = 0
            for i in range(1, len(layers) - 1):
                self.weights.append(2 * random.random([layers[i - 1] + 1, layers[i] + 1]) - 1)
            self.weights.append(2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1)
        else:
            self.load_model(path)

    def train(self, rdd_data, learn_rate=0.5, iteration=100, error=1e-8, method=None):
        if method is None:
            method = self.BP_SGD

        self.logger.debug("Using {} method to do the update".format(method))
        if method == self.BP_SGD:
            self.back_propagation_sgd(rdd_data, learn_rate, iteration, error)

    def back_propagation(self, rdd_data, learn_rate, iteration, error):
        """ Using standard gradient descent to do the back propagation """
        input_data = np.array(rdd_data)
        real = np.array(map(float, input_data[:, 1]))
        feature = input_data[:, 0]
        feature = map(lambda x: map(float, x), feature)
        ones = np.atleast_2d(np.ones(np.shape(feature)[0])) * self.bias
        feature = np.concatenate((ones.T, np.array(feature, dtype=float)), axis=1)
        for i in range(iteration):
            if i % 50 == 0:
                self.logger.debug("Start the {} iteration".format(i))
            k = random.randint(np.shape(feature)[0])
            train_data = feature[k]
            process_data = [np.array(train_data)]
            target = real[k]
            for layer in self.weights:
                activation = self.af(np.dot(process_data[-1], layer))
                process_data.append(activation)

            error = target - process_data[-1]
            deltas = [error * self.afd(process_data[-1])]
            for l in range(len(process_data) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.afd(process_data[l]))

            deltas.reverse()
            for l in range(len(self.weights)):
                layer = np.atleast_2d(process_data[l])
                delta = np.atleast_2d(deltas[l])
                self.weights[l] += learn_rate * layer.T.dot(delta)
            if i % 50 == 0:
                self.logger.debug("{} iteration finished".format(i))

    def back_propagation_sgd(self, rdd_data, learn_rate, iteration, error):
        """ Using stochastic gradient descent to do the back propagation """
        input_data = np.array(rdd_data)
        real = np.array(map(float, input_data[:, 1]))
        feature = input_data[:, 0]
        feature = map(lambda x: map(float, x), feature)
        ones = np.atleast_2d(np.ones(np.shape(feature)[0])) * self.bias
        feature = np.concatenate((ones.T, np.array(feature, dtype=float)), axis=1)
        for i in range(iteration):
            self.logger.debug("Start the {} iteration".format(i))
            k = random.randint(np.shape(feature)[0])
            train_data = feature[k]
            process_data = [np.array(train_data)]
            target = real[k]
            for layer in self.weights:
                activation = self.af(np.dot(process_data[-1], layer))
                process_data.append(activation)

            error = target - process_data[-1]
            deltas = [error * self.afd(process_data[-1])]
            for l in range(len(process_data) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.afd(process_data[l]))

            deltas.reverse()
            for l in range(len(self.weights)):
                layer = np.atleast_2d(process_data[l])
                delta = np.atleast_2d(deltas[l])
                self.weights[l] += learn_rate * layer.T.dot(delta)
            self.logger.debug("{} iteration finished".format(i))

    def predict(self, features):
        temp_feature = np.concatenate((np.ones(1).T * self.bias, np.array(features)))

        # self.logger.debug("features are %s", features)
        for i in range(len(self.weights) - 1):
            temp_feature = self.af(np.dot(temp_feature, self.weights[i]))
        temp_feature = np.dot(temp_feature, self.weights[-1])[0]

        # self.logger.debug("Predict value are %s", temp_feature)
        return temp_feature


def test_nn():
    import os
    data_file = os.path.join(os.path.abspath('../../data'), "0051.HK.csv")

    from StockSimulator.parse_data import DataParser
    from StockSimulator.constant import NONE_DISTRIBUTED
    data = DataParser(path=data_file, window_size=3)
    data_list = data.load_data_from_yahoo_csv()
    close_train_data, close_test_data, open_train_data, open_test_data = \
        data.get_n_days_history_data(data_list, data_type=NONE_DISTRIBUTED, normalized=True)

    close_train_data = map(lambda (x, y): [x, sigmoid(y)], close_train_data)
    neural = NeuralNetwork([4, 5, 1], seed=1234, activation_func=sigmoid, activation_func_dot=sigmoid_dot, bias=1)
    neural.train(rdd_data=close_train_data, learn_rate=1e-3, error=1e-8, iteration=100)
    print neural.weights
    predict_list = map(lambda (x, y): (DataParser.de_normalize(neural.predict(x), x), y), close_test_data)
    import pprint
    pprint.pprint(predict_list)
    mse = sum(map(lambda (x, y): (x - y)**2, predict_list)) / len(predict_list)
    print mse

if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    test_nn()
    # test = NeuralNetwork([2, 3, 1], seed=1234, bias=0.5)
    # y = np.array(
    #     [5.0, 0.0, 10.0, 1.9, 2.3, 19.0, 2.0, 3.0, 4.0, 3.9, -5.0, -10.0, -4.0, 4.0, -0.5, -1.0, 0.0, 0.8, 0.6, 4.0,
    #      7.0, 0.8, 1.0, 0])
    # a = np.mean(y)
    # b = np.std(y)
    # y = (y - np.mean(y)) / np.std(y)
    # y = map(sigmoid, y)
    # x = [[1, 1], [0, 0], [2, 2], [0.5, 0.3], [0.4, 0.5], [8, 1], [1, 0],
    #      [0, 1], [0.5, 1], [0.3, 1.1], [-1, -1], [-2, -2], [1, -2], [-1, 2],
    #      [1, -0.5], [1, -1], [-1.5, 1], [-1.1, 1], [-1.2, 1], [-4, 4],
    #      [2, 1], [0.1, 0.2], [0.2, 0.2], [-0.3, 0.2]]
    # # print zip(x, y)
    #
    # test.train(zip(x, y), learn_rate=1e-3, iteration=10000)
    # c = test.predict(np.array([1.0, 2.0]))
    # print c * b + a
