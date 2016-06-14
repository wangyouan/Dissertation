#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: inference_system
# Author: Mark Wang
# Date: 1/6/2016

import os
import sys
import time
import datetime

from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LinearRegressionWithSGD
from pandas.tseries.offsets import CustomBusinessDay

from StockInference.constant import Constants
from StockInference.DataCollection.data_collect import DataCollect
from StockInference.Regression.distributed_neural_network import NeuralNetworkSpark
from StockInference.util.data_parse import min_max_de_normalize, get_MSE, get_MAD, get_MAPE
from StockInference.DataParser.data_parser import DataParser
from StockInference.util.date_parser import get_ahead_date
from StockInference.util.file_operation import load_data_from_file, save_data_to_file
from StockInference.util.hongkong_calendar import HongKongCalendar

interest_rate_path = "interest_rate"

if sys.platform == 'darwin':
    interest_rate_path = os.path.join('..', interest_rate_path)


class InferenceSystem(Constants):
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        conf = SparkConf()
        conf.setAppName("StockInference")
        self.sc = SparkContext.getOrCreate(conf=conf)
        self.train_data = None
        self.test_data = None
        self.test_data_features = None
        self.total_data_num = 0
        self.data_parser = None
        self.date_list = []
        log4jLogger = self.sc._jvm.org.apache.log4j
        self.logger = log4jLogger.LogManager.getLogger(self.__class__.__name__)

    def get_train_test_data(self, train_test_ratio, start_date, end_date, features=None, data_file_path=None):

        self.logger.info('#################################################################')
        self.logger.info('Get train and testing data')
        self.logger.info('Training / Testing ratio is {}'.format(train_test_ratio))
        self.logger.info('Start date is {}, end date is {}'.format(start_date, end_date))
        self.logger.info('#################################################################')

        # collect data, will do some preliminary process to stock process
        required_info = {
            self.PRICE_TYPE: self.STOCK_CLOSE,
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
            self.FUNDAMENTAL_ANALYSIS: [
                self.US10Y_BOND, self.US30Y_BOND, self.FXI,
                # self.IC, self.IA, # comment this  two because this two bond is a little newer
                self.HSI,
                {self.FROM: self.USD, self.TO: self.HKD},
                {self.FROM: self.EUR, self.TO: self.HKD},
                # {self.FROM: self.AUD, self.TO: self.HKD},
                self.ONE_YEAR, self.HALF_YEAR, self.OVER_NIGHT,
                self.GOLDEN_PRICE,
            ]
        }
        data_collection = DataCollect(stock_symbol=self.stock_symbol, start_date=start_date, end_date=end_date,
                                      data_file_path=data_file_path, logger=self.sc._jvm.org.apache.log4j.LogManager)
        data_collection.set_interest_rate_path(interest_rate_path)
        if features is None:
            features = required_info

        # features = {self.FUNDAMENTAL_ANALYSIS: [self.ONE_YEAR]}
        self.logger.info("No previous data, will collected them from Internet")
        raw_data = data_collection.get_raw_data(label_info=features[self.PRICE_TYPE], required_info=features)

        # debug
        # raw_data_file = open(os.path.join('../output', "raw.dat"), 'w')
        # import pickle
        # pickle.dump(raw_data, raw_data_file)
        # raw_data_file.close()
        #
        # f = open('text.csv', 'w')
        # f.write(
        #     'date,open,high,low,close,macd1,macd2,sma_3,sma_13,sma_21,ema_5,ema_13,ema_21,roc_13,roc_21,rsi_9,rsi_14,rsi_21,us10y,us30y,fxi,hsi,usdhkd,eurhkd,oneyear,halfyear,overnight,golden_price\n')
        # date_list = data_collection.get_date_list()
        # for i in range(len(raw_data)):
        #     f.write("{},{}\n".format(date_list[i], ','.join(map(str, raw_data[i].features))))
        # f.close()
        # raise ValueError("Warn SB")

        # Split train and test
        if self.data_parser is None:
            n_components = None
            self.data_parser = DataParser(n_components=n_components)

            self.train_data, self.test_data, self.test_data_features = self.data_parser.split_train_test_data(
                train_ratio=train_test_ratio, raw_data=raw_data, fit_transform=True)
        else:
            self.train_data, self.test_data, self.test_data_features = self.data_parser.split_train_test_data(
                train_ratio=train_test_ratio, raw_data=raw_data, fit_transform=False)
        self.total_data_num = len(raw_data)
        self.date_list = data_collection.get_date_list()
        self.logger.info('#################################################################')
        self.logger.info('Get train and testing data finished')
        self.logger.info('#################################################################')

    def predict_historical_data(self, train_test_ratio, start_date, end_date, data_folder_path=None,
                                training_method=None, features=None, output_file_path=None, load_model=False):

        """ Get raw data -> process data -> pca -> normalization -> train -> test """
        self.logger.info('Start to predict stock symbol {}'.format(self.stock_symbol))

        if training_method is None:
            training_method = self.ARTIFICIAL_NEURAL_NETWORK

        self.logger.info("The training method is {}".format(training_method))

        if output_file_path is not None and not os.path.isdir(output_file_path):
            os.makedirs(output_file_path)

        if data_folder_path is not None and not os.path.isdir(data_folder_path):
            os.makedirs(data_folder_path)

        if self.train_data is None or self.test_data is None or self.test_data_features is None:
            self.get_train_test_data(train_test_ratio, start_date=start_date, end_date=end_date, features=features,
                                     data_file_path=data_folder_path)

        if output_file_path is not None:
            save_data_to_file(os.path.join(output_file_path, "data_parser.dat"), self.data_parser)

        training_data = self.sc.parallelize(self.train_data)
        testing_data = self.sc.parallelize(self.test_data)
        testing_data_features = self.sc.parallelize(self.test_data_features)

        self.logger.info('#################################################################')
        self.logger.info('Start to training data, the training method is {}'.format(training_method))
        self.logger.info('#################################################################')

        if training_method == self.ARTIFICIAL_NEURAL_NETWORK:
            input_num = len(self.train_data[0].features)

            if output_file_path is not None:
                model_path = os.path.join(output_file_path, 'ann_model.dat')
            else:
                model_path = None

            if output_file_path is not None and load_model:
                self.logger.info("load model from {}".format(output_file_path))
                layer_path = os.path.join(output_file_path, 'layers.dat')
                if os.path.isfile(layer_path):
                    layers = load_data_from_file(layer_path)
                    if layers[0] != input_num:
                        layers = [input_num, input_num / 3 * 2, input_num / 3, 1]
                else:
                    layers = [input_num, input_num / 3 * 2, input_num / 3, 1]

                if os.path.isfile(model_path):
                    model = load_data_from_file(model_path)
                else:
                    model = None

            else:

                # training
                layers = [input_num, input_num / 3 * 2, input_num / 3, 1]
                self.logger.info('Input layer is {}'.format(layers))
                model = None
                if output_file_path:
                    layer_file = open(os.path.join(output_file_path, "layers.txt"), 'w')
                    layer_file.write(str(layers))
                    layer_file.close()
                    save_data_to_file(os.path.join(output_file_path, 'layers.dat'), layers)

            if model is None:
                neural_network = NeuralNetworkSpark(layers=layers, bias=0)
                model = neural_network.train(training_data, method=neural_network.BP, seed=1234, learn_rate=0.0001,
                                             iteration=20, model=model)
                if output_file_path:
                    model.save_model(model_path)
        elif training_method == self.RANDOM_FOREST:

            model = RandomForest.trainRegressor(training_data, categoricalFeaturesInfo={}, numTrees=4,
                                                featureSubsetStrategy="auto", impurity='variance', maxDepth=5,
                                                maxBins=32, seed=1234)

        elif training_method == self.LINEAR_REGRESSION:
            model = LinearRegressionWithSGD.train(training_data, iterations=10000, step=0.001)

        else:
            self.logger.error("Unknown training method {}".format(training_method))
            raise ValueError("Unknown training method {}".format(training_method))

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

        if output_file_path:
            predict_list = predict.collect()
            predict_file = open(os.path.join(output_file_path, "predict_result.csv"), "w")
            predict_file.write("date,origin,predict\n")
            test_date_list = test_date_list[1:]
            test_date_list.append(get_ahead_date(test_date_list[-1], -1))
            for i in range(self.total_data_num - train_data_num):
                predict_file.write("%s,%2f,%2f\n" % (
                    test_date_list[i], predict_list[i][0], predict_list[i][1]))
            predict_file.close()

        return predict

    def get_future_stock_price(self, training_method=None, start_history=None, model_path=None, features=None,
                               output_file_path=None, data_file_path=None):
        today = datetime.datetime.today()
        cday = CustomBusinessDay(calendar=HongKongCalendar(today.year - 1, today.year))
        if today.hour > 18:
            if today.weekday() < 5:
                end_day = today
            else:
                end_day = today - cday
            predict_date = today + cday
        else:
            end_day = today - cday
            if today.weekday() < 5:
                predict_date = today
            else:
                predict_date = today + cday

        predict_date = predict_date.strftime("%Y-%m-%d")
        end_date = end_day.strftime("%Y-%m-%d")
        if model_path is not None:
            model = load_data_from_file(model_path)
        else:

            if start_history is None:
                start_date = datetime.datetime(end_day.year - 2, end_day.month, end_day.day)
                start_date += cday
            else:
                start_date = start_history

            if not isinstance(start_date, str):
                start_date = start_date.strftime("%Y-%m-%d")

            model = self.predict_historical_data(1, start_date, end_date, output_file_path=output_file_path,
                                                 training_method=training_method, data_folder_path=data_file_path,
                                                 features=features, load_model=False)
        data_collection = DataCollect(self.stock_symbol, end_date, end_date, data_file_path=data_file_path,
                                      logger=self.sc._jvm.org.apache.log4j.LogManager)
        data_collection.set_interest_rate_path(interest_rate_path)
        data = data_collection.get_raw_data(features[self.PRICE_TYPE], required_info=features)
        predict_features = self.data_parser.transform(data)[0]
        predict_price = model.predict(predict_features)

        return predict_date, min_max_de_normalize(predict_price, features=data[0])


if __name__ == "__main__":

    output_path = 'output'
    data_path = 'data'

    if sys.platform == 'darwin':
        output_path = '../{}'.format(output_path)
        data_path = '../{}'.format(data_path)

    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    stock_list = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK', '0006.HK', '0007.HK', '0008.HK', '0009.HK',
                  '0010.HK', '0011.HK', '0012.HK', '0013.HK', '0014.HK', '0015.HK', '0016.HK', '0017.HK', '0018.HK',
                  '0019.HK', '0020.HK', '0021.HK', '0022.HK', '0023.HK', '0024.HK', '0025.HK', '0026.HK', '0027.HK',
                  '0028.HK', '0029.HK', '0030.HK', '0031.HK', '0032.HK', '0700.HK', '0034.HK', '0035.HK', '0036.HK',
                  '0068.HK', '0038.HK', '0039.HK', '0040.HK', '0041.HK', '0042.HK', '0043.HK', '0044.HK', '0045.HK',
                  '0046.HK', '0088.HK', '0050.HK', '0051.HK', '0052.HK', '0053.HK', '0054.HK', '0168.HK', '0056.HK',
                  '0057.HK', '0058.HK', '0059.HK', '0060.HK', '0888.HK', '0062.HK', '0063.HK', '0064.HK', '0065.HK',
                  '0066.HK', '1123.HK']

    for method in [InferenceSystem.ARTIFICIAL_NEURAL_NETWORK, InferenceSystem.RANDOM_FOREST,
                   InferenceSystem.LINEAR_REGRESSION][:1]:

        new_file_path = os.path.join(output_path, method)
        if not os.path.isdir(new_file_path):
            os.makedirs(new_file_path)

        f = open(os.path.join(new_file_path, "stock_info.csv"), 'w')
        f.write('stock,MSE,MAPE,MAD\n')
        for stock in stock_list:
            # for stock in ["0033.HK"]:
            specific_file_path = os.path.join(new_file_path, stock[:4])
            test = InferenceSystem(stock)
            predict_result = test.predict_historical_data(0.8, "2006-04-14", "2016-04-15",
                                                          training_method=method,
                                                          data_folder_path=data_path,
                                                          output_file_path=specific_file_path,
                                                          load_model=False)
            mse = get_MSE(predict_result)
            mape = get_MAPE(predict_result)
            mad = get_MAD(predict_result)
            f.write('{},{},{},{}\n'.format(stock, mse, mape, mad))
            test.sc.stop()
            time.sleep(30)

        f.close()
