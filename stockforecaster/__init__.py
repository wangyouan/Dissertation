#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: __init__.py
# Author: Mark Wang
# Date: 24/7/2016

import logging

import numpy as np
import pandas as pd
from pyspark import SparkContext, SparkConf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from stockforecaster.constant import Constants
from stockforecaster.datacollect import DataCollect


class StockForecaster(Constants):
    def __init__(self, stock_symbol, data_path=None, train_method=None, train_system='Spark'):
        if train_method is None:
            self._training_method = self.RANDOM_FOREST
        elif isinstance(train_method, list) or isinstance(train_method, tuple):
            self._training_method = {self.DIRECTION_PREDICTION: train_method[0],
                                     self.AMOUNT_PREDICTION: train_method[1]}
        elif isinstance(train_method, dict):
            self._training_method = train_method
        else:
            self._training_method = train_method

        self._stock_symbol = stock_symbol

        self._train_system = train_system

        if train_system == 'Spark':
            conf = SparkConf()
            conf.setAppName("{}_{}_{}".format(self.__class__.__name__, stock_symbol, train_method))
            self._sc = SparkContext.getOrCreate(conf=conf)

            logger = self._sc._jvm.org.apache.log4j
            self.logger = logger.LogManager.getLogger(self.__class__.__name__)
        else:
            logger = logging
            self.logger = logging.getLogger(self.__class__.__name__)

        self._data_collect = DataCollect(stock_symbol, logger=logger, data_path=data_path)

    def _predict_stock_price(self, model, input_data):
        pass

    def _train_model(self, training_data):
        pass

    def _process_data(self, input_data, test_start_data):
        train = input_data[input_data.index <= test_start_data]
        test = input_data[input_data.index > test_start_data]

        tomorrow_price = input_data[self.STOCK_CLOSE].shift(-1)
        train_mean = train.mean()

        # Replace NaN with mean value
        for key in train.keys():
            train[key] = train[key].replace(np.nan, train_mean[key])
            test[key] = test[key].replace(np.nan, train_mean[key])

        # do normalization
        transformer = MinMaxScaler(feature_range=(-1, 1))
        train_tran = transformer.fit_transform(train)
        test_tran = transformer.transform(test)
        train = pd.DataFrame(train_tran, index=train.index, columns=train.keys())
        test = pd.DataFrame(test_tran, index=test.index, columns=test.keys())

        # add tomorrow price info
        train[self.TARGET_PRICE] = tomorrow_price[tomorrow_price.index <= test_start_data]
        test[self.TARGET_PRICE] = tomorrow_price[tomorrow_price.index > test_start_data]

        return train, test

    def main_process(self, start_date, end_date, test_start_date):
        required_info = {
            self.PRICE_TYPE: self.STOCK_CLOSE,
            self.STOCK_PRICE: {self.DATA_PERIOD: 1},
            self.TECHNICAL_INDICATOR: [
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
                # self.US10Y_BOND, self.US30Y_BOND, self.FXI,
                # self.IC, self.IA, # comment this  two because this two bond is a little newer
                self.HSI, self.SHSE,
                {self.FROM: self.USD, self.TO: self.HKD},
                {self.FROM: self.EUR, self.TO: self.HKD},
                # {self.FROM: self.AUD, self.TO: self.HKD},
                self.ONE_YEAR, self.HALF_YEAR, self.OVER_NIGHT,
                self.GOLDEN_PRICE,
            ]
        }
        data = self._data_collect.get_required_data(required_info=required_info, start_date=start_date,
                                                    end_date=end_date)

        train, test = self._process_data(data, test_start_date)
        model = self._train_model(train)
        result = self._predict_stock_price(model=model, input_data=test)
        return result
