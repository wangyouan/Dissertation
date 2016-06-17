#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: CombineSystem
# Author: Mark Wang
# Date: 16/6/2016

import os
import sys

from StockInference.inference_system import InferenceSystem
from StockInference.DataCollection.data_collect import DataCollect
from StockInference.DataParser.data_parser import DataParser


class MixInferenceSystem(InferenceSystem):
    """ Try to combine two training method together """

    def __init__(self, stock_symbol, features=None, direction_method=None, amount_method=None, output_file_path=None,
                 data_folder_path=None, model_path=None, using_exist_model=False, amount_type=None):
        InferenceSystem.__init__(self, stock_symbol=stock_symbol, data_folder_path=data_folder_path,
                                 features=features, output_file_path=output_file_path, model_path=model_path,
                                 using_exist_model=using_exist_model)
        if direction_method is None:
            self.trend_prediction_method = self.RANDOM_FOREST
        else:
            self.trend_prediction_method = direction_method

        if amount_method is None:
            self.amount_prediction_method = self.ARTIFICIAL_NEURAL_NETWORK
        else:
            self.amount_prediction_method = amount_method

        if amount_type is None:
            self.amount_type = self.RATIO_AMOUNT
        else:
            self.amount_type = amount_type

    def load_parameters(self):
        pass

    def save_parameters(self, model):
        pass

    def train_model(self, model, trend_data, amount_data):
        pass

    def evaluate_model(self, model, testing_data):
        pass

    def get_predict_result(self, model, features):
        pass

    def prepare_data(self, start_date, end_date):

        self.logger.info('Get train and testing data')
        self.logger.info('Start date is {}, end date is {}'.format(start_date, end_date))

        # collect data, will do some preliminary process to stock process
        required_info = {
            self.PRICE_TYPE: self.STOCK_CLOSE,
            self.STOCK_PRICE: {self.DATA_PERIOD: 1},
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
                                      data_file_path=self.data_path, logger=self.sc._jvm.org.apache.log4j.LogManager)
        interest_rate_path = "interest_rate"

        if sys.platform == 'darwin':
            interest_rate_path = os.path.join('..', interest_rate_path)
        data_collection.set_interest_rate_path(interest_rate_path)

        if self.data_features is None:
            self.data_features = required_info

        self.data_features[self.STOCK_PRICE][self.DATA_PERIOD] = 1

        # features = {self.FUNDAMENTAL_ANALYSIS: [self.ONE_YEAR]}
        self.logger.info("No previous data, will collected them from Internet")
        raw_data = data_collection.get_raw_data(label_info=self.data_features[self.PRICE_TYPE],
                                                required_info=self.data_features)
        self.date_list = data_collection.get_date_list()
        self.total_data_num = len(raw_data)

        return raw_data

    def processing_data(self, input_data, train_test_ratio=0.8):
        if self.data_parser is None:
            self.data_parser = DataParser(label_data_type=self.amount_type)
            train, test, test_features = self.data_parser.split_train_test_data(train_test_ratio, input_data, True)
        else:
            train, test, test_features = self.data_parser.split_train_test_data(train_test_ratio, input_data, False)

        return train[0], train[1], test[0], test[1], test_features

    def mix_inference_system(self, start_date, end_date, train_test_ratio=0.8, iterations=10):
        """ Get raw data -> process data -> pca -> normalization -> train -> test """
        self.logger.info('Start to predict stock symbol {}'.format(self.stock_symbol))
        self.logger.info("The training method is {}".format(self.training_method))

        if self.using_exist_model:
            trend_model, amount_model = self.load_parameters()
        else:
            trend_model, amount_model = None

        # Generate training data
        data_list = self.prepare_data(start_date=start_date, end_date=end_date)
        trend_train, amount_train, trend_test, amount_test, test_features = self.processing_data(data_list,
                                                                                                 train_test_ratio)

        self.logger.info("Initialize Model")
        if not self.using_exist_model:
            trend_model, amount_model = self.initialize_model()

        self.logger.info('Start to training model')
        for i in range(iterations):
            self.logger.info("Epoch {} starts".format(i))
            train, test, test_features = self.randomly_split_data(training_data, ratio=0.8)
            model = self.model_training(train, model)
            self.logger.info("Epoch {} finishes".format(i))

            mse, mape, cdc, mad = self.evaluate_model_performance(model, test, test_features)
            self.logger.info("Current MSE is {:.4f}".format(mse))
            self.logger.info("Current MAD is {:.4f}".format(mad))
            self.logger.info("Current MAPE is {:.4f}%".format(mape))
            self.logger.info("Current CDC is {:.4f}%".format(cdc))

            # Just train random forest tree one time
            if self.training_method == self.RANDOM_FOREST:
                break

        # if train ratio is at that level, means that target want the model file, not the
        if train_test_ratio > 0.99:
            return trend_model, amount_model

        # Data prediction part
        self.logger.info("Start to use the model to predict price")
        testing_data = self.sc.parallelize(self.test_data)
        testing_data_features = self.sc.parallelize(self.test_data_features)
        predict = self.model_prediction(model, testing_data=testing_data, testing_data_features=testing_data_features)

        self.save_data_to_file(predict.collect(), "predict_result.csv", self.SAVE_TYPE_OUTPUT)
        self.save_parameters(model)

        return predict
