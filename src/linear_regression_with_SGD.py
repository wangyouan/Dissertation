#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: linear_regression_with_SGD
# Author: Mark Wang
# Date: 1/4/2016
# TODO: Add one n days version use highest, lowest, average close price, average open price as input

from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD

sc = SparkContext(appName="LinearRegressionPredict")

# Close logger
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org").setLevel(logger.Level.OFF)
logger.LogManager.getLogger("akka").setLevel(logger.Level.OFF)


def calculate_data(path=r'../data/0003.HK.csv'):
    """
    Use linear regression with SGD to predict the stock price
    Input are last day, high, low, open and close price, directly output result
    :param path: Data file path
    :return: None
    """

    # Read date from given file
    f = open(path)
    data_str = f.read()
    f.close()
    data_list = [map(float, i.split(',')[1:5] + i.split(',')[6:7]) for i in data_str.split('\n')[1:]]
    data_list = list(reversed(data_list))
    close_train_list = []
    open_train_list = []
    data_len = len(data_list)

    # Using 90% data as training data, the remaining data as testing data
    train_data_len = int(data_len * 0.9)
    for i in xrange(1, train_data_len):
        close_price = data_list[i + 1][3]
        open_price = data_list[i + 1][0]
        variable = data_list[i]
        close_train_list.append(LabeledPoint(features=variable, label=close_price))
        open_train_list.append(LabeledPoint(features=variable, label=open_price))

    close_train_data = sc.parallelize(close_train_list)
    open_train_data = sc.parallelize(open_train_list)

    # Training model
    close_model = LinearRegressionWithSGD.train(close_train_data, step=0.001, iterations=1000)
    open_model = LinearRegressionWithSGD.train(open_train_data, step=0.001, iterations=2000)

    close_test_data_list = []
    open_test_data_list = []
    for i in xrange(train_data_len, data_len - 1):
        close_price = data_list[i + 1][3]
        open_price = data_list[i + 1][0]
        variable = data_list[i]
        close_test_data_list.append(LabeledPoint(features=variable, label=close_price))
        open_test_data_list.append(LabeledPoint(features=variable, label=open_price))

    close_test_data = sc.parallelize(close_test_data_list)
    open_test_data = sc.parallelize(open_test_data_list)

    # predict close data test
    close_value_predict = close_test_data.map(lambda p: (p.label, close_model.predict(p.features)))
    MSE = close_value_predict.map(lambda (v, p): (v - p) ** 2).reduce(lambda x, y: x + y) / close_value_predict.count()
    print("Close Mean Squared Error = " + str(MSE))
    print "Close Model coefficients:", str(close_model)

    # predict open data test
    open_value_predict = open_test_data.map(lambda p: (p.label, open_model.predict(p.features)))
    MSE = open_value_predict.map(lambda (v, p): (v - p) ** 2).reduce(lambda x, y: x + y) / open_value_predict.count()
    print("Open Mean Squared Error = " + str(MSE))
    print "Open Model coefficients:", str(open_model)


if __name__ == "__main__":
    calculate_data()
    sc.stop()
