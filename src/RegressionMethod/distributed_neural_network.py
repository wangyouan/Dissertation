#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: distributed_neural_network
# Author: Mark Wang
# Date: 8/5/2016


import numpy as np
import numpy.random as np_rand
from pyspark import SparkContext, RDD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import DenseVector

from regression_method import Regression
from src import load_logger
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


class NeuralNetwork(Constants):
    def __init__(self, layers, bias=1.0, act_func=None, act_func_prime=None):
        self.logger = load_logger(self.__class__.__name__)
        if act_func is None:
            self.act_func = sigmoid
            self.act_func_prime = sigmoid_prime
        else:
            self.act_func = act_func
            self.act_func_prime = act_func_prime
        self.layers = layers

        self.bias = bias
        self.spark_contest = SparkContext.getOrCreate()

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

        # print model.weights
        for k in range(iteration):
            if k % 100 == 0:
                self.logger.info("The {} iteration starts".format(k))
            self.logger.debug("Start the {} iteration".format(k))

            process_data = [rdd_data]
            for layer in model.weights:
                activation = process_data[-1].map(lambda v: LabeledPoint(features=np_dot(v.features, layer),
                                                                         label=v.label)).cache()
                process_data.append(activation)

            deltas = [
                process_data[-1].map(lambda v: (v.label - v.features[0]) * model.act_func_prime(v.features)).cache()]
            for l in range(len(process_data) - 2, 0, -1):
                deltas.append(deltas[-1].zip(process_data[l]).map(lambda (d, p): np_dot(d, model.weights[l].T) *
                                                                                 model.act_func_prime(
                                                                                     p.features)).cache())
            deltas.reverse()
            for l in range(len(model.weights)):
                delta = deltas[l].map(np_atleast_2d).zip(process_data[l].map(lambda v: np_atleast_2d(v.features))) \
                    .map(lambda (d, l): l.T.dot(d)).sum()
                # while (delta < error * rdd_data.count()).all():
                #     delta *= 10
                model.weights[l] += learn_rate * delta
            self.logger.debug("{} iteration finished".format(k))
        print model.weights
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

        fraction = float(self.spark_contest.defaultParallelism) / rdd_data.count()
        for k in range(iteration):
            if k % 100 == 0:
                self.logger.info("The {} iteration starts".format(k))
            self.logger.debug("Start the {} iteration".format(k))

            sample_rdd = rdd_data.sample(True, fraction)
            process_data = [sample_rdd]
            for layer in model.weights:
                activation = process_data[-1].map(lambda v: LabeledPoint(features=np_dot(v.features, layer),
                                                                         label=v.label)).cache()
                process_data.append(activation)

            deltas = [
                process_data[-1].map(lambda v: (v.label - v.features[0]) * model.act_func_prime(v.features)).cache()]
            for l in range(len(process_data) - 2, 0, -1):
                deltas.append(deltas[-1].zip(process_data[l]).map(lambda (d, p): np_dot(d, model.weights[l].T) *
                                                                                 model.act_func_prime(
                                                                                     p.features)).cache())
            deltas.reverse()
            for l in range(len(model.weights)):
                delta = deltas[l].map(np_atleast_2d).zip(process_data[l].map(lambda v: np_atleast_2d(v.features)))\
                    .map(lambda (d, l): l.T.dot(d)).sum()
                # while (delta < error * rdd_data.count()).all():
                #     delta *= 10
                model.weights[l] += learn_rate * delta
            self.logger.debug("{} iteration finished".format(k))
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

    neural = NeuralNetwork([4, 5, 1], bias=1)
    model = neural.train(rdd_data=close_train_data, learn_rate=1e-3, error=1e-5, iteration=1000, method=neural.BP)
    predict_result = close_test_data.map(lambda p: (p.label, DataParser.de_normalize(model.predict(p.features),
                                                                                     p.features))).cache()
    mse = DataParser.get_MSE(predict_result)
    print mse


if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    test_distributed_ann()
