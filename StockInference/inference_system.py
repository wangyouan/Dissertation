#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: inference_system
# Author: Mark Wang
# Date: 1/6/2016

from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint

from StockInference.constant import Constants
from StockInference.DataCollection.data_collect import DataCollect
from StockInference.Regression.distributed_neural_network import NeuralNetworkSpark
from StockInference.util.data_parse import min_max_de_normalize, get_MSE
from StockInference.DataParser.data_parser import DataParser


class InferenceSystem(Constants):
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        self.neural_network = None
        conf = SparkConf()
        conf.setAppName("StockInference")
        self.sc = SparkContext(conf=conf)

    def predict_historical_data(self, train_test_ratio, start_date, end_date):
        data_collection = DataCollect(stock_symbol=self.stock_symbol)
        required_info = {
            self.STOCK_PRICE: {self.DATA_PERIOD: 5},
            self.FUNDAMENTAL_ANALYSIS: [self.US10Y_BOND, self.US30Y_BOND, self.FXI, self.IC, self.IA]
        }
        calculated_data, label_list = data_collection.get_all_required_data(start_date=start_date, end_date=end_date,
                                                                            label_info=self.STOCK_CLOSE,
                                                                            normalized_method=self.MIN_MAX,
                                                                            required_info=required_info)

        input_num = len(calculated_data[0].features)
        self.neural_network = NeuralNetworkSpark(layers=[input_num, input_num + 2, input_num - 4, 1])
        total_data_num = len(calculated_data)
        train_data_num = int(train_test_ratio * total_data_num)
        training_rdd = self.sc.parallelize(calculated_data[:train_data_num])
        testing_rdd = calculated_data[train_data_num:]
        test_label_list = label_list[train_data_num:]
        for i in range(total_data_num - train_data_num):
            testing_rdd[i] = LabeledPoint(label=test_label_list[i], features=testing_rdd[i].features)
        testing_rdd = self.sc.parallelize(testing_rdd)
        model = self.neural_network.train(training_rdd, method=self.neural_network.BP, seed=1234, learn_rate=0.001,
                                          iteration=100)
        predict_result = testing_rdd.map(
            lambda p: (p.label, model.predict(p.features), p.features)) \
            .map(lambda p: (p[0], p[1], data_collection.de_normalize_stock_price(p[2]))) \
            .map(lambda p: (p[0], min_max_de_normalize(p[1], p[2]))).cache()
        test_date_list = data_collection.get_date_list()[train_data_num:]
        predict_list = predict_result.collect()
        f = open("test.csv", "w")
        f.write("date,origin,predict\n")
        for i in range(total_data_num - train_data_num):
            f.write("%s,%2f,%2f\n" % (
                test_date_list[i], predict_list[i][0], predict_list[i][1]))
        f.close()
        print get_MSE(predict_result)
        self.sc.stop()

    def predict_historical_data_new_process(self, train_test_ratio, start_date, end_date):
        """ Get raw data -> process data -> pca -> normalization -> train -> test """

        # collect data, will do some preliminary process to stock process
        data_collection = DataCollect(stock_symbol=self.stock_symbol)
        required_info = {
            self.STOCK_PRICE: {self.DATA_PERIOD: 5},
            self.STOCK_INDICATOR: [
                (self.MACD, {self.MACD_FAST_PERIOD:12, self.MACD_SLOW_PERIOD: 26, self.MACD_TIME_PERIOD: 9}),
                (self.MACD, {self.MACD_FAST_PERIOD:7, self.MACD_SLOW_PERIOD: 14, self.MACD_TIME_PERIOD: 9}),
                (self.SMA, 3),
                (self.SMA, 13),
                (self.SMA, 21),
                (self.EMA, 5),
                (self.EMA, 13),
                (self.EMA, 21),
                (self.ROC, 13),
                (self.ROC, 21),
                (self.RSI, 9),
                (self.RSI, 14),
                (self.RSI, 21),
            ],
            self.FUNDAMENTAL_ANALYSIS: [self.US10Y_BOND, self.US30Y_BOND, self.FXI,
                                        # self.IC, self.IA,
                                        self.HSI]
        }
        raw_data = data_collection.get_raw_data(start_date=start_date, end_date=end_date,
                                                label_info=self.STOCK_CLOSE, required_info=required_info)

        print raw_data

        # Split train and test
        data_parser = DataParser()
        n_components = 'mle'
        train_data, test_data, test_data_features = data_parser.split_train_test_data(train_ratio=train_test_ratio,
                                                                                      raw_data=raw_data,
                                                                                      n_components=n_components)

        # training
        input_num = len(train_data.take(1)[0].features)
        if input_num < 6:
            layers = [input_num, input_num + 1, 1]
        else:
            layers = [input_num, input_num - 2, input_num - 4, 1]
        self.neural_network = NeuralNetworkSpark(layers=layers, bias=0)
        model = self.neural_network.train(train_data, method=self.neural_network.BP, seed=1234, learn_rate=0.0001,
                                          iteration=10)

        # predicting
        predict_result = test_data.map(lambda p: (p.label, model.predict(p.features))).zip(test_data_features) \
            .map(lambda (p, v): (p[0], min_max_de_normalize(p[1], v))).cache()

        # for testing only
        total_data_num = len(raw_data)
        train_data_num = train_data.count()
        test_date_list = data_collection.get_date_list()[train_data_num:]
        predict_list = predict_result.collect()
        f = open("test.csv", "w")
        f.write("date,origin,predict\n")
        for i in range(total_data_num - train_data_num):
            f.write("%s,%2f,%2f\n" % (
                test_date_list[i], predict_list[i][0], predict_list[i][1]))
        f.close()
        print get_MSE(predict_result)
        self.sc.stop()


if __name__ == "__main__":
    test = InferenceSystem('0003.HK')
    test.predict_historical_data_new_process(0.8, "2006-04-14", "2016-04-15")
