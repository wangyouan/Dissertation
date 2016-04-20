#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: RandomForestRegression
# Author: Mark Wang
# Date: 20/4/2016

from pyspark.mllib.tree import RandomForest

from constant import *
from parse_data import DataParser
from __init__ import sc


def price_predict(path, windows=5):
    input_data = DataParser(path=path, window_size=windows)
    close_train, close_test, open_train, open_test = input_data.get_n_days_history_data(data_type=LABEL_POINT,
                                                                                        spark_context=sc)

    # handle open data
    open_model = RandomForest.trainRegressor(open_train, categoricalFeaturesInfo={}, numTrees=3,
                                             featureSubsetStrategy="auto", impurity='variance', maxDepth=4, maxBins=32)
    open_prediction = open_model.predict(open_test.map(lambda x: x.features))
    open_label_prediction = open_test.zip(open_prediction).map(lambda (t, p):
                                                               (DataParser.de_normalize(t.label, t.features),
                                                                DataParser.de_normalize(p, t.features)))
    testMSE = DataParser.get_MSE(open_label_prediction)
    testMAPE = DataParser.get_MAPE(open_label_prediction)
    testMAD = DataParser.get_MAD(open_label_prediction)
    print('Open Test Mean Squared Error = {}'.format(testMSE))
    print('Open Test Mean Absolute Deviation = {}'.format(testMAD))
    print('Open Test Mean Absolute Percentage Error = {}%'.format(testMAPE * 100))
    # print('Learned regression forest model:')
    # print(open_model.toDebugString())

    # handle close data
    close_model = RandomForest.trainRegressor(close_train, categoricalFeaturesInfo={}, numTrees=3,
                                              featureSubsetStrategy="auto", impurity='variance', maxDepth=4, maxBins=32)
    close_prediction = close_model.predict(close_test.map(lambda x: x.features))
    close_label_prediction = close_test.zip(close_prediction).map(lambda (t, p):
                                                               (DataParser.de_normalize(t.label, t.features),
                                                                DataParser.de_normalize(p, t.features)))
    testMSE = DataParser.get_MSE(close_label_prediction)
    testMAPE = DataParser.get_MAPE(close_label_prediction)
    testMAD = DataParser.get_MAD(close_label_prediction)
    print('Close Test Mean Squared Error = {}'.format(testMSE))
    print('Close Test Mean Absolute Deviation = {}'.format(testMAD))
    print('Close Test Mean Absolute Percentage Error = {}%'.format(testMAPE * 100))
    # print('Learned regression forest model:')
    # print(close_model.toDebugString())

    # Save and load model
    # model.save(sc, "target/tmp/myRandomForestRegressionModel")
    # sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestRegressionModel")


if __name__ == "__main__":
    import os

    stock_symbol = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK']
    # for symbol in stock_symbol:
    #     print symbol
    #     path = os.path.join(r'../data', '{}.csv'.format(symbol))
    #     price_predict(path)
    #     print
    for i in range(1, 2):
        print "Day {}".format(i)
        path = os.path.join(r'../data', '{}.csv'.format(stock_symbol[0]))
        price_predict(path, i)
        print
    sc.stop()
