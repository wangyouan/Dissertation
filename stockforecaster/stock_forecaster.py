#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: stockforecaster
# Author: Mark Wang
# Date: 18/10/2016

import re
import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pyspark.sql import SparkSession

from stockforecaster.constant import Constants
from stockforecaster.datacollect import DataCollect
from stockforecaster.prediction_system.spark_train_system import SparkTrainingSystem
from stockforecaster.prediction_system.tensorflow_train_system import TensorFlowTrainingSystem


class StockForecaster(Constants):
    def __init__(self, stock_symbol, data_path=None, train_method=None, train_system='Spark', using_percentage=True,
                 worker_num=2):
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

        self._using_percentage = using_percentage

        self._predict_system = None

        if train_system == self.SPARK:
            self.spark_worker_numbers = worker_num
            name = "{}_{}_{}".format(self.__class__.__name__, stock_symbol, train_method)
            spark = SparkSession.builder.master('local[%s]' % self.spark_worker_numbers).appName(name) \
                .config('spark.executor.instances', self.spark_worker_numbers).getOrCreate()

            logger = spark._jvm.org.apache.log4j.LogManager
            self.logger = logger.getLogger(self.__class__.__name__)
            self._predict_system = SparkTrainingSystem(spark, self._train_method)

        elif train_system == self.TENSORFLOW:
            self._predict_system = TensorFlowTrainingSystem(self._train_method)
            logger = logging
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            raise ValueError('Unknown training system {}'.format(train_system))

        self._data_collect = DataCollect(stock_symbol, logger=logger, data_path=data_path)

    def _process_data(self, input_data, test_start_date):
        train = input_data[input_data.index <= test_start_date]
        test = input_data[input_data.index > test_start_date]

        tomorrow_price = input_data[self.STOCK_CLOSE].shift(-1)
        today_price = input_data[self.STOCK_CLOSE]
        change_amount = tomorrow_price - today_price
        change_direction = change_amount.apply(lambda x: np.nan if np.isnan(x) else int(x > 0))

        # using change percentage as change direction
        if self._using_percentage:
            change_amount = change_amount.apply(abs) / today_price
        else:
            change_amount = change_amount.apply(abs)
        train_mean = train.mean()

        key_set = train.keys()

        tech_train = pd.DataFrame()
        tech_test = pd.DataFrame()

        # pattern = r'MACD|SMA|EMA|ROC|RSI|PPO|ADX'
        pattern = r'MACD|ROC|RSI|PPO|ADX'

        # Replace NaN with mean value
        for key in key_set:
            train.loc[:, key] = train[key].replace(np.nan, train_mean[key])
            test.loc[:, key] = test[key].replace(np.nan, train_mean[key])

            if re.findall(pattern=pattern, string=key):
                tech_test[key] = test[key]
                tech_train[key] = train[key]
                del test[key]
                del train[key]

        # do normalization
        pipe = Pipeline([('Standard', StandardScaler()),
                         ('PCA', PCA()),
                         ('MinMax', MinMaxScaler(feature_range=self.feature_range))
                         ])
        train_tran = pipe.fit_transform(train)
        test_tran = pipe.transform(test)
        train = pd.DataFrame(train_tran, index=train.index, columns=map(str, range(train_tran.shape[1])))
        test = pd.DataFrame(test_tran, index=test.index, columns=map(str, range(test_tran.shape[1])))

        tech_key_set = tech_test.keys()

        # for key in tech_key_set:
        #     if key.startswith('EMA') or key.startswith('SMA') or key.startswith('MACD'):
        #         tran = MinMaxScaler(feature_range=self.feature_range)
        #         train[key] = tran.fit_transform(tech_train[key].values.reshape((-1, 1)))
        #         test[key] = tran.transform(tech_test[key].values.reshape((-1, 1)))

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
            self.PRICE_TYPE: self.STOCK_ADJUSTED_CLOSED,
            self.STOCK_PRICE: {self.DATA_PERIOD: 1},
            self.TECHNICAL_INDICATOR: [
                # (self.MACD, {self.MACD_FAST_PERIOD: 12, self.MACD_SLOW_PERIOD: 26, self.MACD_TIME_PERIOD: 9}),
                (self.PPO, {self.PPO_FAST_PERIOD: 12, self.PPO_SLOW_PERIOD: 26}),
                (self.PPO, {self.PPO_FAST_PERIOD: 7, self.PPO_SLOW_PERIOD: 14}),
                # (self.MACD, {self.MACD_FAST_PERIOD: 7, self.MACD_SLOW_PERIOD: 14, self.MACD_TIME_PERIOD: 9}),
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
                (self.ADX, 14),
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
                transformer = MinMaxScaler(feature_range=self.feature_range)
                amount = train[self.CHANGE_AMOUNT].values.reshape(-1, 1)
                label[self.CHANGE_AMOUNT] = transformer.fit_transform(amount)
            else:
                transformer = None
                label[self.CHANGE_AMOUNT] = train[self.CHANGE_AMOUNT]
            label[self.CHANGE_DIRECTION] = train[self.CHANGE_DIRECTION]
            del train[self.CHANGE_AMOUNT]
            del train[self.CHANGE_DIRECTION]

        else:

            transformer = MinMaxScaler(feature_range=self.feature_range)
            label[self.TARGET_PRICE] = transformer.fit_transform(train[self.TARGET_PRICE].values.reshape(-1, 1))
            del train[self.TARGET_PRICE]

        self._predict_system.train(train, label)

        result = self._predict_system.predict(test)

        # restore prediction price to ordinary mode
        def reconstruct_price(row):
            today = row[self.TODAY_PRICE]

            # test using raw data
            change_amount = abs(row['AmountPrediction'])
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
            result['prediction'] = transformer.inverse_transform(result['prediction'].values.reshape(-1, 1))

        return result

    def stop_server(self):
        self._predict_system.stop()
