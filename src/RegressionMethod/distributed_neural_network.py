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
            self.act_func = lambda x: 1.0 / (1 + np.exp(-x))
            self.act_func_prime = lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2
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
        # self.weights = self.spark_contest.parallelize(weights)

    def train(self, rdd_data, learn_rate=1e-3, iteration=100, error=1e-8, method=None):
        if method is None:
            method = self.BP_SGD

        self.logger.debug("Using {} method to update the model".format(method))
        if method == self.BP_SGD:
            self.back_propagation_sgd(rdd_data, learn_rate, iteration)

    def back_propagation_sgd(self, rdd_data, learn_rate, iteration):
        """ Using stochastic gradient descent to do the back propagation """
        rdd_data = rdd_data.map(lambda v: LabeledPoint(features=np.concatenate((np.ones(1).T * self.bias,
                                                                                np.array(v.features))),
                                                       label=self.act_func(v.label)))

        rdd_data.cache()
        sample_fraction = float(self.spark_contest.defaultParallelism) / rdd_data.count()
        for k in range(iteration):
            self.logger.debug("Start the {} iteration".format(k))

            sample_rdd = rdd_data.sample(False, sample_fraction)
            process_data = [sample_rdd]
            for layer in self.weights:
                activation = process_data[-1].map(lambda v: LabeledPoint(features=np.dot(v.features, layer),
                                                                         label=v.label))
                process_data.append(activation)

            deltas = [process_data[-1].map(lambda v: (v.label - v.features) * self.act_func_prime(v.features))]
            for l in range(len(process_data) - 2, 0, -1):
                deltas.append(deltas[-1].zip(process_data[l]).map(lambda (d, p): np.dot(d, self.weights[l].T) *
                                                                                 self.act_func_prime(p.features)))
            deltas.reverse()
            for l in range(len(self.weights)):
                layer = process_data[l].map(lambda v: np.atleast_2d(v.features)).sum() / process_data[l].count()
                delta = deltas[l].map(np.atleast_2d).sum() / deltas[l].count()
                self.weights[l] += learn_rate * layer.T.dot(delta)
            self.logger.debug("{} iteration finished".format(i))

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
    predict_result = close_test_data.map(lambda p: (p.label, DataParser.de_normalize(neural.predict(p.features), p.features)))
    mse = DataParser.get_MSE(predict_result)
    print mse

if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    test_distributed_ann()
