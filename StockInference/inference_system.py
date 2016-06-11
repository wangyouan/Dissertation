#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: inference_system
# Author: Mark Wang
# Date: 1/6/2016

import sys

from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LinearRegressionWithSGD

from StockInference.constant import Constants
from StockInference.DataCollection.data_collect import DataCollect
from StockInference.Regression.distributed_neural_network import NeuralNetworkSpark
from StockInference.util.data_parse import min_max_de_normalize, get_MSE, get_MAD, get_MAPE
from StockInference.DataParser.data_parser import DataParser
from StockInference.util.date_parser import get_ahead_date


iterations = 20
# folder = "output/random_forest_not_adj".format(4, iterations)
folder = "output/ann_{}_layer_not_adj_{}_pca_none".format(4, iterations)

if sys.platform == 'darwin':
    folder = "../{}".format(folder)


class InferenceSystem(Constants):
    def __init__(self, stock_symbol, adjusted=False):
        self.stock_symbol = stock_symbol
        conf = SparkConf()
        conf.setAppName("StockInference")
        self.sc = SparkContext.getOrCreate(conf=conf)
        self.train_data = None
        self.test_data = None
        self.test_data_features = None
        self.total_data_num = 0
        self.date_list = None
        self.adjusted_close = adjusted
        self.model = None
        log4jLogger = self.sc._jvm.org.apache.log4j
        self.logger = log4jLogger.LogManager.getLogger(__name__)

    def get_train_test_data(self, train_test_ratio, start_date, end_date):
        self.logger.info('#################################################################')
        self.logger.info('Get train and testing data')
        self.logger.info('Training / Testing ratio is {}'.format(train_test_ratio))
        self.logger.info('Start date is {}, end date is {}'.format(start_date, end_date))
        self.logger.info('#################################################################')

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
                                        # self.IC, self.IA, # comment this  two because this two bond is a little newer
                                        self.HSI, {self.FROM: self.USD, self.TO: self.HKD},
                                        {self.FROM: self.EUR, self.TO: self.HKD},
                                        {self.FROM: self.AUD, self.TO: self.HKD},
                                        {self.GOLDEN_PRICE: False}]
        }
        raw_data = data_collection.get_raw_data(start_date=start_date, end_date=end_date, using_ratio=True,
                                                using_adj=self.adjusted_close, label_info=self.STOCK_CLOSE,
                                                required_info=required_info)
        # debug
        # raw_data_file = open("test", 'w')
        # import pickle
        # pickle.dump(raw_data, raw_data_file)
        # raw_data_file.close()
        # raise ValueError("Warn SB")

        # Split train and test
        n_components = None
        data_parser = DataParser(n_components=n_components)
        self.train_data, self.test_data, self.test_data_features = data_parser.split_train_test_data(
            train_ratio=train_test_ratio, raw_data=raw_data)
        self.total_data_num = len(raw_data)
        self.date_list = data_collection.get_date_list()
        self.logger.info('#################################################################')
        self.logger.info('Get train and testing data finished')
        self.logger.info('Predict adjusted price is {}'.format("Yes" if self.adjusted_close else "No"))
        self.logger.info('#################################################################')

    def predict_historical_data(self, train_test_ratio, start_date, end_date, save_data=True,
                                training_method=None):
        """ Get raw data -> process data -> pca -> normalization -> train -> test """
        self.logger.info('Start to predict stock symbol {}'.format(self.stock_symbol))

        if training_method is None:
            training_method = self.ARTIFICIAL_NEURAL_NETWORK

        if self.train_data is None or self.test_data is None or self.test_data_features is None:
            self.get_train_test_data(train_test_ratio, start_date=start_date, end_date=end_date)

        training_data = self.sc.parallelize(self.train_data)
        testing_data = self.sc.parallelize(self.test_data)
        testing_data_features = self.sc.parallelize(self.test_data_features)

        self.logger.info('#################################################################')
        self.logger.info('Start to training data, the training method is {}'.format(training_method))
        self.logger.info('#################################################################')

        if training_method == self.ARTIFICIAL_NEURAL_NETWORK:
            # training
            input_num = len(self.train_data[0].features)
            layers = [input_num, input_num / 3 * 2, input_num / 3, 1]
            self.logger.info('Input layer is {}'.format(layers))
            if save_data:
                layer_file = open("{}/layers.txt".format(folder), 'w')
                layer_file.write(str(layers))
                layer_file.close()
            neural_network = NeuralNetworkSpark(layers=layers, bias=0)
            model = neural_network.train(training_data, method=neural_network.BP, seed=1234, learn_rate=0.0001,
                                         iteration=iterations)
        elif training_method == self.RANDOM_FOREST:
            model = RandomForest.trainRegressor(training_data, categoricalFeaturesInfo={}, numTrees=4,
                                                featureSubsetStrategy="auto", impurity='variance', maxDepth=5,
                                                maxBins=32, seed=1234)

        elif training_method == self.LINEAR_REGRESSION:
            model = LinearRegressionWithSGD.train(training_data, iterations=10000, step=0.001)

        else:
            self.logger.error("Unknown training method {}".format(training_method))
            raise ValueError("Unknown training method {}".format(training_method))

        self.model = model
        if train_test_ratio > 0.99:
            return model

        self.logger.info("Start to use the model to predict price")
        if training_method != self.RANDOM_FOREST:
            # predicting
            predict = testing_data.map(lambda p: (p.label, model.predict(p.features))) \
                .zip(testing_data_features) \
                .map(lambda (p, v): (p[0], min_max_de_normalize(p[1], v))).cache()
        else:
            predict = model.predict(testing_data.map(lambda x: x.features))
            predict = testing_data.zip(predict).zip(testing_data_features) \
                .map(lambda (m, n): (m[0].label, min_max_de_normalize(m[1], n))).cache()

        train_data_num = len(self.train_data)
        test_date_list = self.date_list[train_data_num:]

        if save_data:
            predict_list = predict.collect()
            predict_file = open("{}/{}.csv".format(folder, self.stock_symbol), "w")
            predict_file.write("date,origin,predict\n")
            test_date_list = test_date_list[1:]
            test_date_list.append(get_ahead_date(test_date_list[-1], -1))
            for i in range(self.total_data_num - train_data_num):
                predict_file.write("%s,%2f,%2f\n" % (
                    test_date_list[i], predict_list[i][0], predict_list[i][1]))
            predict_file.close()

        return predict

    def get_future_stock_price(self, training_method=None, history_date=None):
        pass


if __name__ == "__main__":
    import os

    if not os.path.isdir(folder):
        os.mkdir(folder)

    f = open('{}/stock_test.csv'.format(folder), 'w')
    f.write('stock,MSE,MAPE,MAD\n')
    stock_list = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK', '0006.HK', '0007.HK', '0008.HK', '0009.HK',
                  '0010.HK', '0011.HK', '0012.HK', '0013.HK', '0014.HK', '0015.HK', '0016.HK', '0017.HK', '0018.HK',
                  '0019.HK', '0020.HK', '0021.HK', '0022.HK', '0023.HK', '0024.HK', '0025.HK', '0026.HK', '0027.HK',
                  '0028.HK', '0029.HK', '0030.HK', '0031.HK', '0032.HK', '0700.HK', '0034.HK', '0035.HK', '0036.HK',
                  '0068.HK', '0038.HK', '0039.HK', '0040.HK', '0041.HK', '0042.HK', '0043.HK', '0044.HK', '0045.HK',
                  '0046.HK', '0088.HK', '0050.HK', '0051.HK', '0052.HK', '0053.HK', '0054.HK', '0168.HK', '0056.HK',
                  '0057.HK', '0058.HK', '0059.HK', '0060.HK', '0888.HK', '0062.HK', '0063.HK', '0064.HK', '0065.HK',
                  '0066.HK', '1123.HK']

    for stock in stock_list[:5]:
        test = InferenceSystem(stock, False)
        predict_result = test.predict_historical_data(0.8, "2006-04-14", "2016-04-15", save_data=True,
                                                      training_method=test.ARTIFICIAL_NEURAL_NETWORK)
        mse = get_MSE(predict_result)
        mape = get_MAPE(predict_result)
        mad = get_MAD(predict_result)
        f.write('{},{},{},{}\n'.format(stock, mse, mape, mad))
        test.sc.stop()

    f.close()
