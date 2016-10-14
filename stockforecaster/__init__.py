#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: __init__.py
# Author: Mark Wang
# Date: 24/7/2016

import logging

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, MultilayerPerceptronClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from regression import CustomRegression

from stockforecaster.constant import Constants
from stockforecaster.datacollect import DataCollect

LINEAR_REGRESSION_ITERATION_TIMES = 100000
RANDOM_FOREST_TREE_NUMBER = 100
RANDOM_FOREST_DEPTH = 30


class StockForecaster(Constants):
    def __init__(self, stock_symbol, data_path=None, train_method=None, train_system='Spark'):
        if train_method is None:
            self._train_method = self.RANDOM_FOREST
        elif isinstance(train_method, list) or isinstance(train_method, tuple):
            self._train_method = {self.DIRECTION_PREDICTION: train_method[0],
                                  self.AMOUNT_PREDICTION: train_method[1]}
        elif isinstance(train_method, dict):
            self._train_method = train_method
        else:
            self._train_method = train_method

        self._stock_symbol = stock_symbol

        self._train_system = train_system

        if train_system == self.SPARK:
            name = "{}_{}_{}".format(self.__class__.__name__, stock_symbol, train_method)
            self._spark = SparkSession.builder.appName(name).getOrCreate()

            logger = self._spark._jvm.org.apache.log4j.LogManager
            self.logger = logger.getLogger(self.__class__.__name__)
        else:
            logger = logging
            self.logger = logging.getLogger(self.__class__.__name__)

        self._data_collect = DataCollect(stock_symbol, logger=logger, data_path=data_path)

    def _predict_stock_price(self, model, features):
        if self._train_system == self.SPARK:
            return self._predict_stock_price_spark(model=model, features=features)

    def _predict_stock_price_spark(self, model, features):
        df = self._prepare_data_spark(features)
        if isinstance(model, dict):
            df = model[self.CHANGE_DIRECTION].transform(df)
            df = model[self.CHANGE_AMOUNT].transform(df)
        else:
            df = model.transform(df)

        pdf = df.toPandas()
        pdf['Date'] = features.index
        return pdf.set_index('Date')

    def _train_model(self, features, label):
        if self._train_system == self.SPARK:
            if isinstance(self._train_method, dict):
                features[self.CHANGE_AMOUNT] = label[self.CHANGE_AMOUNT]
                features[self.CHANGE_DIRECTION] = label[self.CHANGE_DIRECTION]
            else:
                features[self.TARGET_PRICE] = label[self.TARGET_PRICE]
            return self._train_model_spark(data=features)

    def _train_model_spark(self, data):
        df = self._prepare_data_spark(data)
        input_num = len(data.keys()) - 2
        if isinstance(self._train_method, dict):
            model = {self.CHANGE_AMOUNT: None,
                     self.CHANGE_DIRECTION: None}

            if self._train_method[self.CHANGE_AMOUNT] == self.LINEAR_REGRESSION:
                lr = LinearRegression(featuresCol="features", labelCol=self.CHANGE_AMOUNT,
                                      maxIter=LINEAR_REGRESSION_ITERATION_TIMES,
                                      predictionCol='AmountPrediction')
                model[self.CHANGE_AMOUNT] = lr.fit(df)
            elif self._train_method[self.CHANGE_AMOUNT] == self.RANDOM_FOREST:
                rfr = RandomForestRegressor(featuresCol="features", labelCol=self.CHANGE_AMOUNT,
                                            numTrees=RANDOM_FOREST_TREE_NUMBER,
                                            maxDepth=RANDOM_FOREST_DEPTH, predictionCol='AmountPrediction')
                model[self.CHANGE_AMOUNT] = rfr.fit(df)
            elif self._train_method == self.ARTIFICIAL_NEURAL_NETWORK:
                ann = CustomRegression(MLPRegressor(hidden_layer_sizes=(input_num / 3 * 2, input_num / 3),
                                                    learning_rate_init=0.0001, max_iter=1000))
                model = ann.fit(df.select('features').rdd, df.select(self.CHANGE_AMOUNT).rdd)
            else:
                self.logger.warn('Unsupported training method {}'.format(self._train_method))
                raise ValueError('Unsupported training method {}'.format(self._train_method))

            if self._train_method[self.CHANGE_DIRECTION] == self.LOGISTIC_REGRESSION:
                lr = LogisticRegression(featuresCol="features", labelCol=self.CHANGE_DIRECTION,
                                        maxIter=LINEAR_REGRESSION_ITERATION_TIMES,
                                        predictionCol='DirPrediction')
                model[self.CHANGE_DIRECTION] = lr.fit(df)
            elif self._train_method[self.CHANGE_DIRECTION] == self.RANDOM_FOREST:
                rfc = RandomForestClassifier(featuresCol="features", labelCol=self.CHANGE_DIRECTION,
                                             numTrees=RANDOM_FOREST_TREE_NUMBER,
                                             maxDepth=RANDOM_FOREST_DEPTH, predictionCol='DirPrediction')
                model[self.CHANGE_DIRECTION] = rfc.fit(df)

            elif self._train_method[self.CHANGE_DIRECTION] == self.ARTIFICIAL_NEURAL_NETWORK:
                mlpc = MultilayerPerceptronClassifier(featuresCol="features", labelCol=self.CHANGE_DIRECTION,
                                                      layers=[input_num, input_num / 3 * 2, input_num / 3, 1],
                                                      predictionCol='DirPrediction')
                model[self.CHANGE_DIRECTION] = mlpc.fit(df)

            else:
                self.logger.warn('Unsupported training method {}'.format(self._train_method))
                raise ValueError('Unsupported training method {}'.format(self._train_method))

        else:
            if self._train_method == self.LINEAR_REGRESSION:
                lr = LinearRegression(featuresCol="features", labelCol=self.TARGET_PRICE,
                                      maxIter=LINEAR_REGRESSION_ITERATION_TIMES)
                model = lr.fit(df)
            elif self._train_method == self.RANDOM_FOREST:
                rfr = RandomForestRegressor(featuresCol="features", labelCol=self.TARGET_PRICE,
                                            numTrees=RANDOM_FOREST_TREE_NUMBER,
                                            maxDepth=RANDOM_FOREST_DEPTH)
                model = rfr.fit(df)

            elif self._train_method == self.ARTIFICIAL_NEURAL_NETWORK:
                ann = CustomRegression(MLPRegressor(hidden_layer_sizes=(input_num / 3 * 2, input_num / 3),
                                                    learning_rate_init=0.0001, max_iter=1000))
                model = ann.fit(df.select('features').rdd, df.select(self.TARGET_PRICE).rdd)

            else:
                self.logger.warn('Unsupported training method {}'.format(self._train_method))
                raise ValueError('Unsupported training method {}'.format(self._train_method))

        return model

    def _prepare_data_spark(self, data):
        keys = data.keys()
        for key in keys:
            if '.' in key:
                new_key = key.replace('.', '')
                data[new_key] = data[key]
                del data[key]

        keys = list(set(data.keys()).difference({self.CHANGE_AMOUNT, self.CHANGE_DIRECTION, self.TARGET_PRICE,
                                                 self.TODAY_PRICE}))

        df = self._spark.createDataFrame(data)
        ass = VectorAssembler(inputCols=keys, outputCol="features")
        output = ass.transform(df)
        return output

    def _process_data(self, input_data, test_start_data):
        train = input_data[input_data.index <= test_start_data]
        test = input_data[input_data.index > test_start_data]

        tomorrow_price = input_data[self.STOCK_CLOSE].shift(-1)
        today_price = input_data[self.STOCK_CLOSE]
        change_amount = tomorrow_price - today_price
        change_direction = change_amount.apply(lambda x: np.nan if np.isnan(x) else x >= 0)
        train_mean = train.mean()

        # Replace NaN with mean value
        for key in train.keys():
            train.loc[:, key] = train[key].replace(np.nan, train_mean[key])
            test.loc[:, key] = test[key].replace(np.nan, train_mean[key])

        # do normalization
        transformer = MinMaxScaler(feature_range=(-1, 1))
        train_tran = transformer.fit_transform(train)
        test_tran = transformer.transform(test)
        train = pd.DataFrame(train_tran, index=train.index, columns=train.keys())
        test = pd.DataFrame(test_tran, index=test.index, columns=test.keys())

        # add tomorrow price info
        train[self.TARGET_PRICE] = tomorrow_price[tomorrow_price.index <= test_start_data]
        test[self.TARGET_PRICE] = tomorrow_price[tomorrow_price.index > test_start_data]

        if isinstance(self._train_method, dict):
            train[self.CHANGE_AMOUNT] = change_amount[change_amount.index <= test_start_data]
            test[self.CHANGE_AMOUNT] = change_amount[change_amount.index > test_start_data]
            train[self.CHANGE_DIRECTION] = change_direction[change_direction.index <= test_start_data]
            test[self.CHANGE_DIRECTION] = change_direction[change_direction.index > test_start_data]

        train[self.TODAY_PRICE] = today_price[today_price.index <= test_start_data]
        test[self.TODAY_PRICE] = today_price[today_price.index > test_start_data]

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

        transformer = MinMaxScaler(feature_range=(-1, 1))
        label = pd.DataFrame(index=train.index)
        # label_test = pd.DataFrame(index=test.index)

        # prepare result label
        if isinstance(self._train_method, dict):
            amount = train[self.CHANGE_AMOUNT].values.reshape(-1, 1)
            label[self.CHANGE_DIRECTION] = train[self.CHANGE_DIRECTION]
            del train[self.CHANGE_AMOUNT]
            del train[self.CHANGE_DIRECTION]
            label[self.CHANGE_AMOUNT] = transformer.fit_transform(amount)

            # label_test[self.CHANGE_DIRECTION] = test[self.CHANGE_DIRECTION]
            # label_test[self.CHANGE_AMOUNT] = test[self.CHANGE_AMOUNT]
            # del test[self.CHANGE_DIRECTION]
            # del test[self.CHANGE_AMOUNT]

        else:
            label[self.TARGET_PRICE] = transformer.fit_transform(train[self.TARGET_PRICE].values.reshape(-1, 1))
            del train[self.TARGET_PRICE]

            # label_test[self.TARGET_PRICE] = test[self.TARGET_PRICE]
            # del test[self.TARGET_PRICE]

        model = self._train_model(train, label)

        result = self._predict_stock_price(model=model, features=test)

        # restore prediction price to ordinary mode
        def reconstruct_price(row):
            today = row[self.TODAY_PRICE]
            change_amount = row['AmountPrediction']
            direction = row['DirPrediction']
            if direction == 1:
                return today + change_amount
            else:
                return today - change_amount

        if isinstance(self._train_method, dict):
            result['AmountPrediction'] = transformer.inverse_transform(result['AmountPrediction'].values.reshape(-1, 1))

            result['prediction'] = result.apply(reconstruct_price, axis=1)

        else:
            result['prediction'] = transformer.inverse_transform(result['prediction'].values.reshape(-1, 1))

        return result
