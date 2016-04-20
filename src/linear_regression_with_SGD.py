#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: linear_regression_with_SGD
# Author: Mark Wang
# Date: 1/4/2016

import os

from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD

from parse_data import DataParser
from constant import *
from __init__ import sc


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
    data_list = [map(float, i.split(',')[1:5]) for i in data_str.split('\n')[1:]]
    data_list = list(reversed(data_list))
    close_train_list = []
    open_train_list = []
    data_len = len(data_list)

    # Using 90% data as training data, the remaining data as testing data
    train_data_len = int(data_len * 0.8)
    for i in range(1, train_data_len):
        close_price = data_list[i + 1][3]
        open_price = data_list[i + 1][0]
        variable = data_list[i]
        close_train_list.append(LabeledPoint(features=variable, label=close_price))
        open_train_list.append(LabeledPoint(features=variable, label=open_price))

    close_train_data = sc.parallelize(close_train_list)
    open_train_data = sc.parallelize(open_train_list)

    # Training model
    close_model = LinearRegressionWithSGD.train(close_train_data, step=0.001, iterations=1000)
    open_model = LinearRegressionWithSGD.train(open_train_data, step=0.001, iterations=1000)

    close_test_data_list = []
    open_test_data_list = []
    for i in range(train_data_len, data_len - 1):
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
    print("Close Model coefficients:", str(close_model))

    # predict open data test
    open_value_predict = open_test_data.map(lambda p: (p.label, open_model.predict(p.features)))
    MSE = open_value_predict.map(lambda (v, p): (v - p) ** 2).reduce(lambda x, y: x + y) / open_value_predict.count()
    print("Open Mean Squared Error = " + str(MSE))
    print("Open Model coefficients:", str(open_model))


def calculate_data_normalized(path=r'../data/0003.HK.csv', windows=5):
    """
    Use linear regression with SGD to predict the stock price
    Input are last day, high, low, open and close price, directly output result
    :param path: Data file path
    :return: None
    """

    # Read date from given file
    data = DataParser(path=path, window_size=windows)

    data_list = data.load_data_from_yahoo_csv()
    close_train_data, close_test_data, open_train_data, open_test_data = \
        data.get_n_days_history_data(data_list, data_type=LABEL_POINT, spark_context=sc)

    # Training model
    close_model = LinearRegressionWithSGD.train(close_train_data, step=0.0001, iterations=1000)
    open_model = LinearRegressionWithSGD.train(open_train_data, step=0.0001, iterations=1000)

    def de_normalize_label_point(p):
        return p.label * (p.features[1] - p.features[2]) / 2 + (p.features[1] + p.features[2]) / 2

    def de_normalize_data(label, features):
        return label * (features[1] - features[2]) / 2 + (features[1] + features[2]) / 2

    # #de normalized data
    # close_test_data = close_test_data.map(normalize)
    # open_test_data = close_test_data.map(normalize)

    # predict close data test
    close_value_predict = close_test_data.map(lambda p: (de_normalize_label_point(p),
                                                         de_normalize_data(close_model.predict(p.features),
                                                                           p.features)))
    MSE = close_value_predict.map(lambda (v, p): (v - p) ** 2).reduce(lambda x, y: x + y) / close_value_predict.count()
    print("Close Mean Squared Error = " + str(MSE))
    print("Close Model coefficients:", str(close_model))

    # predict open data test
    open_value_predict = open_test_data.map(lambda p: (de_normalize_label_point(p),
                                                       de_normalize_data(close_model.predict(p.features), p.features)))
    MSE = open_value_predict.map(lambda (v, p): (v - p) ** 2).reduce(lambda x, y: x + y) / open_value_predict.count()
    print("Open Mean Squared Error = " + str(MSE))
    print("Open Model coefficients:", str(open_model))


if __name__ == "__main__":
    stock_symbol = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK']
    for symbol in stock_symbol:
        path = os.path.join(r'../data', '{}.csv'.format(symbol))
        print("Non normalize version")
        calculate_data(path)

        print("Normalized version")
        for window in range(1, 2):
            calculate_data_normalized(path, windows=window)
    sc.stop()
