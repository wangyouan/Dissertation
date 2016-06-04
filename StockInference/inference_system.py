#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: inference_system
# Author: Mark Wang
# Date: 1/6/2016

from pyspark import SparkContext, SparkConf

from StockInference.constant import Constants
from StockInference.DataCollection.data_collect import DataCollect
from StockInference.Regression.distributed_neural_network import NeuralNetworkSpark
from StockInference.util.data_parse import de_normalize, get_MSE


class InferenceSystem(Constants):
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        self.data_collection = DataCollect(stock_symbol=stock_symbol)
        self.neural_network = None
        conf = SparkConf()
        conf.setAppName("StockInference")
        self.sc = SparkContext(conf=conf)

    def predict_historical_data(self, train_test_ratio, start_date, end_date):
        required_info = {
            self.STOCK_PRICE: {self.DATA_PERIOD: 5},
            self.FUNDAMENTAL_ANALYSIS: [self.US10Y_BOND, self.US30Y_BOND, self.FXI, self.IC, self.IA]
        }
        calculated_data = self.data_collection.get_all_required_data(start_date=start_date, end_date=end_date,
                                                                     label_info=self.STOCK_CLOSE,
                                                                     normalized_method=self.MIN_MAX,
                                                                     required_info=required_info)
        input_num = len(calculated_data[0].features)
        self.neural_network = NeuralNetworkSpark(layers=[input_num, input_num + 2, input_num - 4, 1])
        total_data_num = len(calculated_data)
        train_data_num = int(train_test_ratio * total_data_num)
        training_rdd = self.sc.parallelize(calculated_data[:train_data_num])
        testing_rdd = self.sc.parallelize(calculated_data[train_data_num:])
        model = self.neural_network.train(training_rdd, method=self.neural_network.BP, seed=1234, learn_rate=0.001,
                                          iteration=100)
        predict_result = testing_rdd.map(
            lambda p: (de_normalize(p.label, p.features), de_normalize(model.predict(p.features), p.features)))
        print get_MSE(predict_result)
        self.sc.stop()


if __name__ == "__main__":
    test = InferenceSystem('0003.HK')
    test.predict_historical_data(0.8, "2006-04-14", "2016-04-15")
