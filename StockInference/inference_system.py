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
from StockInference.util.data_parse import min_max_de_normalize, get_MSE, get_MAD, get_MAPE
from StockInference.DataParser.data_parser import DataParser

folder = "gold_true_ratio_ajd_close"


class InferenceSystem(Constants):
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        self.neural_network = None
        conf = SparkConf()
        conf.setAppName("StockInference")
        self.sc = SparkContext.getOrCreate(conf=conf)

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
        predict = testing_rdd.map(
            lambda p: (p.label, model.predict(p.features), p.features)) \
            .map(lambda p: (p[0], p[1], data_collection.de_normalize_stock_price(p[2]))) \
            .map(lambda p: (p[0], min_max_de_normalize(p[1], p[2]))).cache()
        test_date_list = data_collection.get_date_list()[train_data_num:]
        predict_list = predict.collect()
        test_file = open("test.csv", "w")
        test_file.write("date,origin,predict\n")
        for i in range(total_data_num - train_data_num):
            test_file.write("%s,%2f,%2f\n" % (
                test_date_list[i], predict_list[i][0], predict_list[i][1]))
        test_file.close()
        print get_MSE(predict)
        self.sc.stop()

    def predict_historical_data_new_process(self, train_test_ratio, start_date, end_date):
        """ Get raw data -> process data -> pca -> normalization -> train -> test """

        # collect data, will do some preliminary process to stock process
        data_collection = DataCollect(stock_symbol=self.stock_symbol)
        required_info = {
            self.STOCK_PRICE: {self.DATA_PERIOD: 5},
            self.STOCK_INDICATOR: [
                (self.MACD, {self.MACD_FAST_PERIOD: 12, self.MACD_SLOW_PERIOD: 26, self.MACD_TIME_PERIOD: 9}),
                (self.MACD, {self.MACD_FAST_PERIOD: 7, self.MACD_SLOW_PERIOD: 14, self.MACD_TIME_PERIOD: 9}),
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
                                        self.HSI, {self.FROM: self.USD, self.TO: self.HKD},
                                        {self.FROM: self.EUR, self.TO: self.HKD},
                                        {self.FROM: self.AUD, self.TO: self.HKD},
                                        {self.GOLDEN_PRICE: True}]
        }
        raw_data = data_collection.get_raw_data(start_date=start_date, end_date=end_date, using_ratio=True,
                                                using_adj=True, label_info=self.STOCK_CLOSE,
                                                required_info=required_info)

        # print raw_data
        # return
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
        predict = test_data.map(lambda p: (p.label, model.predict(p.features))).zip(test_data_features) \
            .map(lambda (p, v): (p[0], min_max_de_normalize(p[1], v))).cache()

        # for testing only
        total_data_num = len(raw_data)
        train_data_num = train_data.count()
        test_date_list = data_collection.get_date_list()[train_data_num:]
        predict_list = predict.collect()
        predict_file = open("../output/{}/{}.csv".format(folder, self.stock_symbol), "w")
        predict_file.write("date,origin,predict\n")
        test_date_list = test_date_list[1:]
        test_date_list.append(data_collection.get_ahead_date(test_date_list[-1], -1))
        for i in range(total_data_num - train_data_num):
            predict_file.write("%s,%2f,%2f\n" % (
                test_date_list[i], predict_list[i][0], predict_list[i][1]))
        predict_file.close()
        return predict


if __name__ == "__main__":
    import os

    if not os.path.isdir('../output/{}'.format(folder)):
        os.mkdir('../output/{}'.format(folder))

    f = open('../output/{}/stock_test.csv'.format(folder), 'w')
    f.write('stock,MSE,MAPE,MAD\n')
    stock_list = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK', '0006.HK', '0007.HK', '0008.HK', '0009.HK',
                  '0010.HK', '0011.HK', '0012.HK', '0013.HK', '0014.HK', '0015.HK', '0016.HK', '0017.HK', '0018.HK',
                  '0019.HK', '0020.HK', '0021.HK', '0022.HK', '0023.HK', '0024.HK', '0025.HK', '0026.HK', '0027.HK',
                  '0028.HK', '0029.HK', '0030.HK', '0031.HK', '0032.HK', '0033.HK', '0034.HK', '0035.HK', '0036.HK',
                  '0037.HK', '0038.HK', '0039.HK', '0040.HK', '0041.HK', '0042.HK', '0043.HK', '0044.HK', '0045.HK',
                  '0046.HK', '0048.HK', '0050.HK', '0051.HK', '0052.HK', '0053.HK', '0054.HK', '0055.HK', '0056.HK',
                  '0057.HK', '0058.HK', '0059.HK', '0060.HK', '0061.HK', '0062.HK', '0063.HK', '0064.HK', '0065.HK',
                  '0066.HK', '1123.HK']

    # stock_list = ['0700.HK']
    for stock in stock_list[:1]:
        test = InferenceSystem(stock)
        predict_result = test.predict_historical_data_new_process(0.8, "2006-04-14", "2016-04-15")
        mse = get_MSE(predict_result)
        mape = get_MAPE(predict_result)
        mad = get_MAD(predict_result)
        f.write('{},{},{},{}\n'.format(stock, mse, mape, mad))
        test.sc.stop()

    f.close()
