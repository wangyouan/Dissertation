#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: __init__.py
# Author: Mark Wang
# Date: 24/7/2016

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pyspark.sql import SparkSession

from stockforecaster.constant import Constants
from stockforecaster.datacollect import DataCollect
from stockforecaster.regression_method.neural_network_regression_spark import KerasNeuralNetworkSpark
from stockforecaster.prediction_system.train_with_spark import SparkTrainingSystem

LINEAR_REGRESSION_ITERATION_TIMES = 100000
RANDOM_FOREST_TREE_NUMBER = 30
RANDOM_FOREST_DEPTH = 20
WORKER_NUMBERS = 2


class StockForecaster(Constants):
    def __init__(self, stock_symbol, data_path=None, train_method=None, train_system='Spark', using_percentage=True):
        if train_method is None:
            self._train_method = self.RANDOM_FOREST
        elif isinstance(train_method, list) or isinstance(train_method, tuple):
            self._train_method = {self.CHANGE_DIRECTION: train_method[0],
                                  self.CHANGE_AMOUNT: train_method[1]}
        elif isinstance(train_method, dict):
            self._train_method = train_method
        else:
            self._train_method = train_method

        self._stock_symbol = stock_symbol

        self._train_system = train_system

        self._using_percentage = using_percentage

        self._predict_system = None

        if train_system == self.SPARK:
            name = "{}_{}_{}".format(self.__class__.__name__, stock_symbol, train_method)
            spark = SparkSession.builder.appName(name).getOrCreate()

            logger = spark._jvm.org.apache.log4j.LogManager
            self.logger = logger.getLogger(self.__class__.__name__)
            self._predict_system = SparkTrainingSystem(spark, self._train_method)
        else:
            logger = logging
            self.logger = logging.getLogger(self.__class__.__name__)

        self._data_collect = DataCollect(stock_symbol, logger=logger, data_path=data_path)

    def _process_data(self, input_data, test_start_date):
        train = input_data[input_data.index <= test_start_date]
        test = input_data[input_data.index > test_start_date]

        tomorrow_price = input_data[self.STOCK_CLOSE].shift(-1)
        today_price = input_data[self.STOCK_CLOSE]
        change_amount = tomorrow_price - today_price
        change_direction = change_amount.apply(lambda x: np.nan if np.isnan(x) else int(x >= 0))

        # using change percentage as change direction
        if self._using_percentage:
            change_amount = change_amount.apply(abs) / today_price
        else:
            change_amount = change_amount.apply(abs)
        train_mean = train.mean()

        # Replace NaN with mean value
        for key in train.keys():
            train.loc[:, key] = train[key].replace(np.nan, train_mean[key])
            test.loc[:, key] = test[key].replace(np.nan, train_mean[key])

        # do normalization
        pipe = Pipeline([('Standard', StandardScaler()),
                         ('PCA', PCA()),
                         ('MinMax', MinMaxScaler(feature_range=(-1, 1)))
                         ])
        train_tran = pipe.fit_transform(train)
        test_tran = pipe.transform(test)
        train = pd.DataFrame(train_tran, index=train.index, columns=map(str, range(train_tran.shape[1])))
        test = pd.DataFrame(test_tran, index=test.index, columns=map(str, range(test_tran.shape[1])))

        # add tomorrow price info
        train[self.TARGET_PRICE] = tomorrow_price[tomorrow_price.index <= test_start_date]
        test[self.TARGET_PRICE] = tomorrow_price[tomorrow_price.index > test_start_date]

        if isinstance(self._train_method, dict):
            train[self.CHANGE_AMOUNT] = change_amount[change_amount.index <= test_start_date]
            test[self.CHANGE_AMOUNT] = change_amount[change_amount.index > test_start_date]
            train[self.CHANGE_DIRECTION] = change_direction[change_direction.index <= test_start_date]
            test[self.CHANGE_DIRECTION] = change_direction[change_direction.index > test_start_date]

        train[self.TODAY_PRICE] = today_price[today_price.index <= test_start_date]
        test[self.TODAY_PRICE] = today_price[today_price.index > test_start_date]

        test = test.dropna(how='any')

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
        label = pd.DataFrame(index=train.index)
        # label_test = pd.DataFrame(index=test.index)

        # prepare result label
        if isinstance(self._train_method, dict):

            if not self._using_percentage:
                transformer = MinMaxScaler(feature_range=(-1, 1))
                amount = train[self.CHANGE_AMOUNT].values.reshape(-1, 1)
                label[self.CHANGE_AMOUNT] = transformer.fit_transform(amount)
            else:
                transformer = None
                label[self.CHANGE_AMOUNT] = train[self.CHANGE_AMOUNT]
            label[self.CHANGE_DIRECTION] = train[self.CHANGE_DIRECTION]
            del train[self.CHANGE_AMOUNT]
            del train[self.CHANGE_DIRECTION]

        else:

            transformer = MinMaxScaler(feature_range=(-1, 1))
            label[self.TARGET_PRICE] = transformer.fit_transform(train[self.TARGET_PRICE].values.reshape(-1, 1))
            del train[self.TARGET_PRICE]

        self._predict_system.train(train, label)

        result = self._predict_system.predict(test)

        # restore prediction price to ordinary mode
        def reconstruct_price(row):
            today = row[self.TODAY_PRICE]
            change_amount = row['AmountPrediction']
            direction = row['DirPrediction']
            if self._using_percentage:
                if direction == 1:
                    return today * (1 + change_amount)
                else:
                    return today * (1 - change_amount)
            else:
                if direction == 1:
                    return today + change_amount
                else:
                    return today - change_amount

        if isinstance(self._train_method, dict):
            if not self._using_percentage:
                result['AmountPrediction'] = transformer.inverse_transform(result['AmountPrediction']
                                                                           .values.reshape(-1, 1))

            result['prediction'] = result.apply(reconstruct_price, axis=1)

        else:
            if not self._using_percentage:
                result['prediction'] = transformer.inverse_transform(result['prediction'].values.reshape(-1, 1))

        return result
