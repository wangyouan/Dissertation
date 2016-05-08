#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: linear_regression_with_SGD
# Author: Mark Wang
# Date: 1/4/2016

import os

from __init__ import load_spark_context, load_logger
from constant import *
from parse_data import DataParser
from plot_data import plot_label_vs_data
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD

non_normalize_mad = []
non_normalize_mse = []
non_normalize_mape = []

normalize_mad = []
normalize_mse = []
normalize_mape = []

logger = load_logger(application_name="LinearRegression")

def calculate_data(path=r'../data/0003.HK.csv', sc=None):
    """
    Use linear regression with SGD to predict the stock price
    Input are last day, high, low, open and close price, directly output result
    :param path: Data file path
    :return: None
    """
    logger.debug("input file is {}".format(path.split('/')[-1]))
    if sc is None:
        logger.info("Start spark context")
        sc = load_spark_context()[0]
        logger.info("Load spark successfully")

    # Read date from given file
    logger.debug("Load file")
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
    logger.debug("Prepare train date")
    logger.debug("Total input data: {}".format(data_len))
    logger.debug("Total train data: {}".format(train_data_len))
    for i in range(1, train_data_len):
        close_price = data_list[i + 1][3]
        open_price = data_list[i + 1][0]
        variable = data_list[i]
        close_train_list.append(LabeledPoint(features=variable, label=close_price))
        open_train_list.append(LabeledPoint(features=variable, label=open_price))

    close_train_data = sc.parallelize(close_train_list)
    open_train_data = sc.parallelize(open_train_list)

    # Training model
    logger.debug("Start training")
    close_model = LinearRegressionWithSGD.train(close_train_data, step=0.001, iterations=1000)
    open_model = LinearRegressionWithSGD.train(open_train_data, step=0.001, iterations=1000)

    logger.debug("Prepare test data")
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
    logger.debug("Calculate close predict data")
    close_value_predict = close_test_data.map(lambda p: (p.label, close_model.predict(p.features)))
    MSE = close_value_predict.map(lambda (v, p): (v - p) ** 2).reduce(lambda x, y: x + y) / close_value_predict.count()
    MAD = DataParser.get_MAD(close_value_predict)
    MAPE = DataParser.get_MAPE(close_value_predict)
    logger.info("Close Mean Squared Error = " + str(MSE))
    logger.info("Close Mean Absolute Deviation = " + str(MAD))
    logger.info("Close Mean Absolute Percentage Error = " + str(MAPE))
    logger.info("Close Model coefficients:", str(close_model))

    # predict open data test
    logger.debug("Calculate open predict data")
    open_value_predict = open_test_data.map(lambda p: (p.label, open_model.predict(p.features)))
    MSE = open_value_predict.map(lambda (v, p): (v - p) ** 2).reduce(lambda x, y: x + y) / open_value_predict.count()
    MAD = DataParser.get_MAD(close_value_predict)
    MAPE = DataParser.get_MAPE(close_value_predict)
    non_normalize_mad.append(MAD)
    non_normalize_mape.append(MAPE)
    non_normalize_mse.append(MSE)
    logger.info("Open Mean Squared Error = " + str(MSE))
    logger.info("Open Mean Absolute Deviation = " + str(MAD))
    logger.info("Open Mean Absolute Percentage Error = " + str(MAPE))
    logger.info("Open Model coefficients:", str(open_model))
    return close_value_predict, open_value_predict


def calculate_data_non_normalized(path=r'../data/0003.HK.csv', windows=5, spark_context=None):
    """
    Use linear regression with SGD to predict the stock price
    Input are last day, high, low, open and close price, directly output result
    :param path: Data file path
    :return: None
    """
    if spark_context is None:
        spark_context = load_spark_context()[0]

    # Read date from given file
    data = DataParser(path=path, window_size=windows)

    data_list = data.load_data_from_yahoo_csv()
    close_train_data, close_test_data, open_train_data, open_test_data = \
        data.get_n_days_history_data(data_list, data_type=LABEL_POINT, spark_context=spark_context, normalized=False)

    # Training model
    close_model = LinearRegressionWithSGD.train(close_train_data, step=0.0001, iterations=1000)
    open_model = LinearRegressionWithSGD.train(open_train_data, step=0.0001, iterations=1000)

    # predict close data test
    logger.debug("Calculate close predict data")
    close_value_predict = close_test_data.map(lambda p: (p.label, close_model.predict(p.features)))
    MSE = close_value_predict.map(lambda (v, p): (v - p) ** 2).reduce(lambda x, y: x + y) / close_value_predict.count()
    MAD = DataParser.get_MAD(close_value_predict)
    MAPE = DataParser.get_MAPE(close_value_predict)
    logger.info("Close Mean Squared Error = " + str(MSE))
    logger.info("Close Mean Absolute Deviation = " + str(MAD))
    logger.info("Close Mean Absolute Percentage Error = " + str(MAPE))
    logger.info("Close Model coefficients:", str(close_model))

    # predict open data test
    logger.debug("Calculate open predict data")
    open_value_predict = open_test_data.map(lambda p: (p.label, close_model.predict(p.features)))
    MSE = open_value_predict.map(lambda (v, p): (v - p) ** 2).reduce(lambda x, y: x + y) / open_value_predict.count()
    MAD = DataParser.get_MAD(open_value_predict)
    MAPE = DataParser.get_MAPE(open_value_predict)
    non_normalize_mad.append(MAD)
    non_normalize_mape.append(MAPE)
    non_normalize_mse.append(MSE)
    logger.info("Open Mean Squared Error = " + str(MSE))
    logger.info("Open Mean Absolute Deviation = " + str(MAD))
    logger.info("Open Mean Absolute Percentage Error = " + str(MAPE))
    logger.info("Open Model coefficients:", str(open_model))
    return close_value_predict, open_value_predict


def calculate_data_normalized(path=r'../data/0003.HK.csv', windows=5, spark_context=None):
    """
    Use linear regression with SGD to predict the stock price
    Input are last day, high, low, open and close price, directly output result
    :param path: Data file path
    :return: None
    """
    logger.info("Start to calculate data with normalized using Linear regression")
    if spark_context is None:
        spark_context = load_spark_context()[0]

    # Read date from given file
    logger.debug("Load data")
    data = DataParser(path=path, window_size=windows)

    data_list = data.load_data_from_yahoo_csv()
    close_train_data, close_test_data, open_train_data, open_test_data = \
        data.get_n_days_history_data(data_list, data_type=LABEL_POINT, spark_context=spark_context)

    # Training model
    logger.debug("Start training")
    close_model = LinearRegressionWithSGD.train(close_train_data, step=0.0001, iterations=100000)
    open_model = LinearRegressionWithSGD.train(open_train_data, step=0.0001, iterations=100000)

    def de_normalize_data(label, features):
        return label * (features[1] - features[2]) / 2 + (features[1] + features[2]) / 2

    # predict close data test
    logger.debug("Calculate close predict data")
    close_value_predict = close_test_data.map(lambda p: (p.label, de_normalize_data(close_model.predict(p.features),
                                                                              p.features)))
    MSE = close_value_predict.map(lambda (v, p): (v - p) ** 2).reduce(lambda x, y: x + y) / close_value_predict.count()
    MAD = DataParser.get_MAD(close_value_predict)
    MAPE = DataParser.get_MAPE(close_value_predict)
    logger.debug("Close Mean Squared Error = " + str(MSE))
    logger.debug("Close Mean Absolute Deviation = " + str(MAD))
    logger.debug("Close Mean Absolute Percentage Error = " + str(MAPE))
    logger.debug("Close Model coefficients:", str(close_model))

    # predict open data test
    logger.debug("Calculate open predict data")
    open_value_predict = open_test_data.map(
        lambda p: (p.label, de_normalize_data(close_model.predict(p.features), p.features)))
    MSE = open_value_predict.map(lambda (v, p): (v - p) ** 2).reduce(lambda x, y: x + y) / open_value_predict.count()
    MAD = DataParser.get_MAD(open_value_predict)
    MAPE = DataParser.get_MAPE(open_value_predict)
    normalize_mad.append(MAD)
    normalize_mape.append(MAPE)
    normalize_mse.append(MSE)
    logger.debug("Open Mean Squared Error = " + str(MSE))
    logger.debug("Open Mean Absolute Deviation = " + str(MAD))
    logger.debug("Open Mean Absolute Percentage Error = " + str(MAPE))
    logger.debug("Open Model coefficients:", str(open_model))
    logger.info("Stock price calculate succeed")
    return close_value_predict, open_value_predict


def test_non_vs_normalize(show_plt=False, windows=10, stock_num=None):
    sc = load_spark_context()[0]
    if show_plt:
        import matplotlib.pyplot as plt
    else:
        plt = None

    file_list = os.listdir(r'../data')
    symbol_list = []

    if stock_num is None:
        stock_num = len(file_list)

    index = 0
    for symbol in file_list:
        if not stock_num:
            break
        print symbol
        path = os.path.join(r'../data', symbol)
        # if not symbol.startswith('00') or not os.path.isfile(path):
        if '0002' not in path and '0003' not in path:
            continue
        print("Non normalize version")
        non_normalize = calculate_data_non_normalized(path, windows=windows, spark_context=sc)
        print("Normalized version")
        normalize = calculate_data_normalized(path, windows=windows, spark_context=sc)
        symbol_list.append(symbol)

        if show_plt:
            open_data = normalize[0].zip(non_normalize[0]).map(lambda (v, p): (v[0], v[1], p[1])).take(100)
            close_data = normalize[1].zip(non_normalize[1]).map(lambda (v, p): (v[0], v[1], p[1])).take(100)
            plot_label_vs_data(data=close_data, label=["real", "Normalized", "Non-Normalized"], graph_index=index,
                               graph_title="close price compare", plt=plt)
            plot_label_vs_data(data=open_data, label=["real", "Normalized", "Non-Normalized"], graph_index=index + 1,
                               graph_title="open price compare", plt=plt)
            index += 2

        stock_num -= 1

    print non_normalize_mse
    print non_normalize_mape
    print non_normalize_mad

    print normalize_mse
    print normalize_mape
    print normalize_mad

    print symbol_list
    if show_plt:
        plt.show()
    sc.stop()


if __name__ == "__main__":
    test_non_vs_normalize(windows=5, stock_num=None, show_plt=True)
