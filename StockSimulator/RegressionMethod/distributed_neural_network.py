#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: distributed_neural_network
# Author: Mark Wang
# Date: 8/5/2016

import logging

import numpy as np
import numpy.random as np_rand
from pyspark import SparkContext, RDD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import DenseVector

from StockSimulator.RegressionMethod.regression_method import Regression
from constants import Constants


def sigmoid(x):
    import numpy as np
    if isinstance(x, DenseVector):
        x = x.toArray()
    return 1.0 / (1 + np.exp(-x))


def sigmoid_prime(x):
    import numpy as np
    if isinstance(x, DenseVector):
        x = x.toArray()
    y = np.exp(-x)
    return y / (1 + y) ** 2


class NeuralNetworkSpark(Constants):
    def __init__(self, layers, bias=1.0, act_func=None, act_func_prime=None):
        if act_func is None:
            self.act_func = sigmoid
            self.act_func_prime = sigmoid_prime
        else:
            self.act_func = act_func
            self.act_func_prime = act_func_prime
        self.layers = layers

        self.bias = bias
        self.spark_context = SparkContext.getOrCreate()

        log4jLogger = self.spark_context._jvm.org.apache.log4j
        self.logger = log4jLogger.LogManager.getLogger(__name__)

    def train(self, rdd_data, learn_rate=1e-3, iteration=100, error=1e-8, method=None, model=None):
        if model is None:
            model = NeuralNetworkModel(self.layers, seed=1234, bias=self.bias, act_func_set=self.act_func,
                                       act_func_prime_set=self.act_func_prime)
        if method is None:
            method = self.BP_SGD

        self.logger.debug("Using {} method to update the model".format(method))
        if method == self.BP_SGD:
            model = self.back_propagation_sgd(rdd_data, learn_rate, iteration, model, error)
        elif method == self.BP:
            model = self.back_propagation(rdd_data=rdd_data, learn_rate=learn_rate, iteration=iteration, model=model,
                                          error=error)
        return model

    def back_propagation(self, rdd_data, learn_rate, iteration, model, error):
        """ Using standard gradient descent method to correct error """

        # define some functions used in the map process
        ones = np.ones(1).T * self.bias
        np_array = np.array
        concatenate = np.concatenate
        np_dot = np.dot
        np_atleast_2d = np.atleast_2d
        rdd_data = rdd_data.map(lambda v: LabeledPoint(features=concatenate((ones, np_array(v.features))),
                                                       label=model.act_func(v.label))).cache()

        data_num = rdd_data.count()

        # print model.weights
        for k in range(iteration):
            self.logger.info("Start the {} iteration".format(k))

            process_data = [rdd_data]
            for layer in model.weights:
                activation = process_data[-1].map(lambda v: LabeledPoint(features=np_dot(v.features, layer),
                                                                         label=v.label)).cache()
                process_data.append(activation)

            # deltas = [
            #     process_data[-1].map(lambda v: (v.label - v.features[0]) * model.act_func_prime(v.features)).cache()]
            # for l in range(len(process_data) - 2, 0, -1):
            #     deltas.append(deltas[-1].zip(process_data[l]).map(lambda (d, p): np_dot(d, model.weights[l].T) *
            #                                                                      model.act_func_prime(
            #                                                                          p.features)).cache())
            # deltas.reverse()
            # for l in range(len(model.weights)):
            #     delta = deltas[l].map(np_atleast_2d).zip(process_data[l].map(lambda v: np_atleast_2d(v.features))) \
            #         .map(lambda (d, l): l.T.dot(d)).sum() / rdd_data.count()
            #     model.weights[l] += learn_rate * delta

            # Update weights
            deltas = process_data[-1].map(lambda v: (v.label - v.features[0]) * model.act_func_prime(v.features))
            delta = deltas.map(np_atleast_2d).zip(process_data[-1].map(lambda v: np_atleast_2d(v.features))) \
                        .map(lambda (d, l): l.T.dot(d)).sum() / data_num
            model.weights[-1] += learn_rate * delta
            for l in range(len(process_data) - 2, 0, -1):
                delta = deltas.map(np_atleast_2d).zip(process_data[l].map(lambda v: np_atleast_2d(v.features))) \
                    .map(lambda (d, l): l.T.dot(d)).sum()
                deltas = deltas.zip(process_data[l]).map(
                    lambda (d, p): np_dot(d, model.weights[l].T) * model.act_func_prime(p.features))
                model.weights[l] += learn_rate * delta / data_num

            self.logger.info("{} iteration finished".format(k))
        self.logger.info("\n{}".format(model.weights))
        return model

    def back_propagation_sgd(self, rdd_data, learn_rate, iteration, model, error):
        """ Using stochastic gradient descent to do the back propagation """

        # define some functions used in the map process
        ones = np.ones(1).T * self.bias
        np_array = np.array
        concatenate = np.concatenate
        np_dot = np.dot
        np_atleast_2d = np.atleast_2d
        rdd_data = rdd_data.map(lambda v: LabeledPoint(features=concatenate((ones, np_array(v.features))),
                                                       label=model.act_func(v.label))).cache()

        fraction = float(self.spark_context.defaultParallelism) / rdd_data.count()
        for k in range(iteration):
            self.logger.info("Start the {} iteration".format(k))

            sample_rdd = rdd_data.sample(True, fraction).cache()
            if sample_rdd.count() == 0:
                continue
            process_data = [sample_rdd]
            for layer in model.weights:
                activation = process_data[-1].map(lambda v: LabeledPoint(features=np_dot(v.features, layer),
                                                                         label=v.label)).cache()
                process_data.append(activation)

            # deltas = [
            #     process_data[-1].map(lambda v: (v.label - v.features[0]) * model.act_func_prime(v.features)).cache()]
            # for l in range(len(process_data) - 2, 0, -1):
            #     deltas.append(deltas[-1].zip(process_data[l]).map(lambda (d, p): np_dot(d, model.weights[l].T) *
            #                                                                      model.act_func_prime(
            #                                                                          p.features)).cache())
            # deltas.reverse()
            # for l in range(len(model.weights)):
            #     delta = deltas[l].map(np_atleast_2d).zip(process_data[l].map(lambda v: np_atleast_2d(v.features))) \
            #                 .map(lambda (d, l): l.T.dot(d)).sum() / sample_rdd.count()
            #     model.weights[l] += learn_rate * delta

            deltas = process_data[-1].map(lambda v: (v.label - v.features[0]) * model.act_func_prime(v.features))
            delta = deltas.map(np_atleast_2d).zip(process_data[-1].map(lambda v: np_atleast_2d(v.features))) \
                        .map(lambda (d, l): l.T.dot(d)).sum()
            model.weights[-1] += learn_rate * delta
            for l in range(len(process_data) - 2, 0, -1):
                delta = deltas.map(np_atleast_2d).zip(process_data[l].map(lambda v: np_atleast_2d(v.features))) \
                    .map(lambda (d, l): l.T.dot(d)).sum()
                deltas = deltas.zip(process_data[l]).map(
                    lambda (d, p): np_dot(d, model.weights[l].T) * model.act_func_prime(p.features))
                model.weights[l] += learn_rate * delta
            self.logger.info("{} iteration finished".format(k))
        self.logger.info("\n{}".format(model.weights))
        return model


class NeuralNetworkModel(Regression):
    def __init__(self, layers, seed=None, bias=1.0, act_func_set=None, act_func_prime_set=None):
        Regression.__init__(self)
        del self.logger
        if act_func_set is None:
            self.act_func = sigmoid
            self.act_func_prime = sigmoid_prime
        else:
            self.act_func = act_func_set
            self.act_func_prime = act_func_prime_set

        self.bias = bias
        self.weights = []

        if seed is not None:
            np_rand.seed(seed=seed)

        i = 0
        for i in range(1, len(layers) - 1):
            self.weights.append(2 * np_rand.random([layers[i - 1] + 1, layers[i] + 1]) - 1)
        self.weights.append(2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1)

    def predict(self, features):

        if isinstance(features, RDD):
            return features.map(self.predict)

        import numpy as np
        temp_feature = np.concatenate((np.ones(1).T * self.bias, np.array(features)))

        for i in range(len(self.weights) - 1):
            temp_feature = self.act_func(np.dot(temp_feature, self.weights[i]))
        temp_feature = np.dot(temp_feature, self.weights[-1])[0]

        return temp_feature
