#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: distributed_neural_network
# Author: Mark Wang
# Date: 8/5/2016


import numpy as np
import numpy.random as np_rand
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint

from regression_method import Regression


class NeuralNetwork(Regression):
    def __init__(self, layers, seed=None, bias=1.0, act_func=None, act_func_prime=None):
        Regression.__init__(self)
        if act_func is None:
            np_exp = np.exp
            self.act_func = lambda x: 1.0 / (1 + np_exp(-x, None))
            self.act_func_prime = lambda x: np_exp(-x, None) / (1 + np_exp(-x, None)) ** 2
        else:
            self.act_func = act_func
            self.act_func_prime = act_func_prime

        self.bias = bias
        self.spark_contest = SparkContext.getOrCreate()

        self.logger.debug("Init weights")
        self.weights = []

        if seed is not None:
            np_rand.seed(seed=seed)

        i = 0
        for i in range(1, len(layers) - 1):
            self.weights.append(2 * np_rand.random([layers[i - 1] + 1, layers[i] + 1]) - 1)
        self.weights.append(2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1)

    def train(self, rdd_data, learn_rate=1e-3, iteration=100, error=1e-8, method=None):
        if method is None:
            method = self.BP_SGD

        self.logger.debug("Using {} method to update the model".format(method))
        if method == self.BP_SGD:
            self.back_propagation_sgd(rdd_data, learn_rate, iteration)

    def back_propagation_sgd(self, rdd_data, learn_rate, iteration):
        """ Using stochastic gradient descent to do the back propagation """

        # define some functions used in the map process
        ones = np.ones(1).T * self.bias
        np_array = np.array
        act_func = self.act_func
        act_func_prime = self.act_func_prime
        concatenate = np.concatenate
        np_dot = np.dot
        np_atleast_2d = np.atleast_2d
        rdd_data = rdd_data.map(lambda v: LabeledPoint(features=concatenate((ones, np_array(v.features))),
                                                       label=act_func(v.label))).cache()
        weights = self.weights

        sample_fraction = float(self.spark_contest.defaultParallelism) / rdd_data.count()
        for k in range(iteration):
            self.logger.debug("Start the {} iteration".format(k))

            sample_rdd = rdd_data.sample(False, sample_fraction)
            process_data = [sample_rdd.cache()]
            for layer in self.weights:
                activation = process_data[-1].map(lambda v: LabeledPoint(features=np_dot(v.features, layer),
                                                                         label=v.label)).cache()
                process_data.append(activation)

            deltas = [process_data[-1].map(lambda v: (v.label - v.features) * act_func_prime(v.features)).cache()]
            for l in range(len(process_data) - 2, 0, -1):
                deltas.append(deltas[-1].zip(process_data[l]).map(lambda (d, p): np_dot(d, weights[l].T) *
                                                                                 act_func_prime(p.features)).cache())
            deltas.reverse()
            for l in range(len(self.weights)):
                layer = process_data[l].map(lambda v: np_atleast_2d(v.features)).sum() / process_data[l].count()
                delta = deltas[l].map(np_atleast_2d).sum() / deltas[l].count()
                self.weights[l] += learn_rate * layer.T.dot(delta)
            self.logger.debug("{} iteration finished".format(k))

    def predict(self, features):
        temp_feature = np.concatenate((np.ones(1).T * self.bias, np.array(features)))

        self.logger.debug("features are %s", features)
        for i in range(len(self.weights) - 1):
            temp_feature = self.act_func(np.dot(temp_feature, self.weights[i]))
        temp_feature = np.dot(temp_feature, self.weights[-1])[0]

        self.logger.debug("Predict value are %s", temp_feature)
        return temp_feature


def test_distributed_ann():
    import os
    data_file = os.path.join(os.path.abspath('../../data'), "0051.HK.csv")

    from src.parse_data import DataParser
    from src.constant import LABEL_POINT
    from src import load_spark_context
    sc = load_spark_context("NeuralNetwork")[0]

    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

    data = DataParser(path=data_file, window_size=3)
    data_list = data.load_data_from_yahoo_csv()
    close_train_data, close_test_data, open_train_data, open_test_data = \
        data.get_n_days_history_data(data_list, data_type=LABEL_POINT, normalized=True, spark_context=sc)

    neural = NeuralNetwork([4, 5, 1], seed=1234, bias=1)
    neural.train(rdd_data=close_train_data, learn_rate=1e-3, error=1e-8, iteration=10000)
    predict_result = close_test_data.map(lambda p: (p.label, DataParser.de_normalize(neural.predict(p.features),
                                                                                     p.features)))
    mse = DataParser.get_MSE(predict_result)
    print mse


if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    test_distributed_ann()
