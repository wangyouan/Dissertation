#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: RandomForestRegression
# Author: Mark Wang
# Date: 20/4/2016

from pyspark import SparkContext
from pyspark.mllib.tree import RandomForest

from constant import *
from parse_data import DataParser

mad_list = []
mape_list = []
mse_list = []


def price_predict(path, windows=5):
    spark_context = SparkContext.getOrCreate()
    log4jLogger = spark_context._jvm.org.apache.log4j
    logger = log4jLogger.LogManager.getLogger(__name__)
    logger.info("Start to predict data using Random Forest")

    input_data = DataParser(path=path, window_size=windows)
    close_train, close_test, open_train, open_test = input_data.get_n_days_history_data(data_type=LABEL_POINT,
                                                                                        spark_context=spark_context)

    # handle open data
    open_model = RandomForest.trainRegressor(open_train, categoricalFeaturesInfo={}, numTrees=3,
                                             featureSubsetStrategy="auto", impurity='variance', maxDepth=4, maxBins=32)
    open_prediction = open_model.predict(open_test.map(lambda x: x.features))
    open_label_prediction = open_test.zip(open_prediction).map(lambda (t, p):
                                                               (t.label,
                                                                DataParser.de_normalize(p, t.features)))
    testMSE = DataParser.get_MSE(open_label_prediction)
    testMAPE = DataParser.get_MAPE(open_label_prediction)
    testMAD = DataParser.get_MAD(open_label_prediction)
    logger.debug('Open Test Mean Squared Error = {}'.format(testMSE))
    logger.debug('Open Test Mean Absolute Deviation = {}'.format(testMAD))
    logger.debug('Open Test Mean Absolute Percentage Error = {}%'.format(testMAPE * 100))
    # print('Learned regression forest model:')
    # print(open_model.toDebugString())

    mad_list.append(testMAD)
    mse_list.append(testMSE)
    mape_list.append(testMAPE)

    # handle close data
    close_model = RandomForest.trainRegressor(close_train, categoricalFeaturesInfo={}, numTrees=3,
                                              featureSubsetStrategy="auto", impurity='variance', maxDepth=4, maxBins=32)
    close_prediction = close_model.predict(close_test.map(lambda x: x.features))
    close_label_prediction = close_test.zip(close_prediction).map(lambda (t, p):
                                                                  (t.label,
                                                                   DataParser.de_normalize(p, t.features)))
    testMSE = DataParser.get_MSE(close_label_prediction)
    testMAPE = DataParser.get_MAPE(close_label_prediction)
    testMAD = DataParser.get_MAD(close_label_prediction)
    logger.debug('Close Test Mean Squared Error = {}'.format(testMSE))
    logger.debug('Close Test Mean Absolute Deviation = {}'.format(testMAD))
    logger.debug('Close Test Mean Absolute Percentage Error = {}%'.format(testMAPE * 100))
    # print('Learned regression forest model:')
    # print(close_model.toDebugString())

    # Save and load model
    # model.save(spark_context, "target/tmp/myRandomForestRegressionModel")
    # sameModel = RandomForestModel.load(spark_context, "target/tmp/myRandomForestRegressionModel")
    return close_label_prediction, open_label_prediction


if __name__ == "__main__":
    import os
    import numpy as np

    stock_symbol = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK']

    # import matplotlib.pyplot as plt
    index = 0
    open_list = []
    for i in range(3, 12):
        print "Day {}".format(i)
        path = os.path.join(r'../data', '{}.csv'.format(stock_symbol[0]))
        close_price, open_price = price_predict(path, i)
        open_price = np.array(open_price.take(100)).T
        if i == 3:
            open_list.append(open_price[0])
        open_list.append(open_price[1])
        index += 2

    print open_list
    print mape_list
    print mse_list
    print mad_list
    # f = open('days_comp.csv', 'w')
    #
    # f.close()
    # plt.show()
    sc.stop()
