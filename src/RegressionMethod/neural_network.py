#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: neural_network
# Author: Mark Wang
# Date: 5/5/2016

import numpy.random as random
import numpy as np

from regression_method import Regression


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
    def __init__(self, layers, seed=None, path=None, activation_func=None, activation_func_dot=None):
        Regression.__init__(self)
        if activation_func is None:
            self.af = sigmoid
            self.afd = sigmoid_dot
        else:
            self.af = activation_func
            self.afd = activation_func_dot

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
            method = self.BP

        self.logger.debug("Using {} method to do the update".format(method))
        if method == self.BP:
            self.back_propagation(rdd_data, learn_rate, iteration, error)

    def back_propagation_sgd(self, rdd_data, learn_rate, iteration, error):
        """ Using stochastic gradient descent to do the back propagation """
        for i in range(iteration):
            self.logger.debug("Start the {} iteration".format(i))
            k = random.randint(len(rdd_data))
            train_data = rdd_data[k]
            x = train_data[0][:]
            x.insert(0, 1)
            process_data = [np.array(x)]
            target = train_data[1]
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

    def back_propagation(self, rdd_data, learn_rate, iteration, error):
        """ Standard gradient descent version """
        input_data = np.array(rdd_data)
        target = np.array(map(float, input_data[:,1]))
        feature = input_data[:, 0]
        for i in range(iteration):
            self.logger.debug("Start the {} iteration".format(i))

            process_data = [np.array(x)]
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
        temp_feature = np.concatenate((np.ones(1).T, np.array(features)))

        self.logger.debug("features are %s", features)
        for weights in self.weights:
            temp_feature = self.af(np.dot(temp_feature, weights))

        self.logger.debug("Predict value are %s", temp_feature)
        return temp_feature[0]


if __name__ == "__main__":
    from src import load_spark_context
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    test = NeuralNetwork([2, 3, 1], seed=1234)
    y = np.array(
        [5.0, 0.0, 10.0, 1.9, 2.3, 19.0, 2.0, 3.0, 4.0, 3.9, -5.0, -10.0, -4.0, 4.0, -0.5, -1.0, 0.0, 0.8, 0.6, 4.0,
         7.0, 0.8])
    a = np.mean(y)
    b = np.std(y)
    y = (y - np.mean(y)) / np.std(y)
    y = map(sigmoid, y)
    x = [[1, 1], [0, 0], [2, 2], [0.5, 0.3], [0.4, 0.5], [8, 1], [1, 0],
         [0, 1], [0.5, 1], [0.3, 1.1], [-1, -1], [-2, -2], [1, -2], [-1, 2],
         [1, -0.5], [1, -1], [-1.5, 1], [-1.1, 1], [-1.2, 1], [-4, 4],
         [2, 1], [0.1, 0.2]]
    test.train(zip(x, y), learn_rate=1e-3, iteration=10000)
    c = test.predict(np.array([1.0, 1.0]))
    c = de_sigmoid(c)
    print c * b + a
