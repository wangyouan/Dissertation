#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: CombineSystem
# Author: Mark Wang
# Date: 16/6/2016

import os
import sys
import shutil

import numpy as np
from pyspark.mllib.regression import LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.classification import LogisticRegressionWithSGD, SVMWithSGD, NaiveBayes, LogisticRegressionModel, \
    SVMModel, NaiveBayesModel

from StockInference.inference_system import InferenceSystem
from StockInference.util.data_parse import *
from StockInference.constant import Constants
from StockInference.DataCollection.data_collect import DataCollect
from StockInference.DataParser.data_parser import DataParser
from StockInference.util.date_parser import string_to_datetime
from StockInference.Regression.distributed_neural_network import NeuralNetworkModel, NeuralNetworkSpark


def get_predict_result(trend_model, amount_model, today_price, features, amount_type):
    amount_prediction = amount_model.predict(features)
    trend_prediction = 1 if trend_model.predict(features) > 0.5 else -1
    if amount_type == Constants.RAW_AMOUNT:
        return today_price + amount_prediction * trend_prediction
    else:
        return today_price * (1 + amount_prediction * trend_prediction)


def get_predict_result_from_data(amount, trend, amount_type, today_price):
    if trend > 0.5:
        trend = 1
    else:
        trend = -1
    if amount_type == Constants.RAW_AMOUNT:
        return today_price + amount * trend
    else:
        return today_price * (1 + amount * trend)


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

        del self.train_data
        del self.training_method
        del self.test_data
        self.train_data_number = 0
        self.trend_model = {'model': None,
                            'cdc': 0}
        self.amount_model = {'model': None,
                             'rmse': float('inf')}

    def load_parameters(self):
        self.amount_prediction_method = self.load_data_from_file(data_type=self.SAVE_TYPE_MODEL,
                                                                 file_name='amount_method')
        self.trend_prediction_method = self.load_data_from_file(data_type=self.SAVE_TYPE_MODEL,
                                                                file_name='trend_method')
        self.data_features = self.load_data_from_file(data_type=self.SAVE_TYPE_MODEL, file_name='features')
        self.stock_symbol = self.load_data_from_file(data_type=self.SAVE_TYPE_MODEL, file_name='symbol')
        self.data_parser = self.load_data_from_file(data_type=self.SAVE_TYPE_MODEL, file_name='data_parser')
        amount_model_path = os.path.join(os.path.abspath(self.model_path), 'amount_model')
        trend_model_path = os.path.join(os.path.abspath(self.model_path), 'trend_model')

        if self.amount_prediction_method == self.RANDOM_FOREST:
            amount_model = RandomForestModel.load(sc=self.sc, path=amount_model_path)
        elif self.amount_prediction_method == self.LINEAR_REGRESSION:
            amount_model = LinearRegressionModel.load(sc=self.sc, path=amount_model_path)
        else:
            amount_model = self.load_data_from_file(data_type=self.SAVE_TYPE_MODEL, file_name='amount_model')

        if self.trend_prediction_method == self.RANDOM_FOREST:
            trend_model = RandomForestModel.load(sc=self.sc, path=trend_model_path)
        elif self.trend_prediction_method == self.LOGISTIC_REGRESSION:
            trend_model = LogisticRegressionModel.load(sc=self.sc, path=trend_model_path)
        elif self.trend_prediction_method == self.NAIVE_BAYES:
            trend_model = NaiveBayesModel.load(sc=self.sc, path=trend_model_path)
        elif self.trend_prediction_method == self.SVM:
            trend_model = SVMModel.load(sc=self.sc, path=trend_model_path)
        else:
            trend_model = self.load_data_from_file(data_type=self.SAVE_TYPE_MODEL, file_name='trend_model')

        return trend_model, amount_model

    def save_model(self, trend_model, amount_model):
        self.save_data_to_file(self.amount_prediction_method, file_name='amount_method', data_type=self.SAVE_TYPE_MODEL)
        self.save_data_to_file(self.trend_prediction_method, file_name='trend_method', data_type=self.SAVE_TYPE_MODEL)
        self.save_data_to_file(self.data_features, file_name='features', data_type=self.SAVE_TYPE_MODEL)
        self.save_data_to_file(self.stock_symbol, file_name='symbol', data_type=self.SAVE_TYPE_MODEL)
        self.save_data_to_file(self.data_parser, file_name='data_parser', data_type=self.SAVE_TYPE_MODEL)

        # save amount prediction model
        path = os.path.join(os.path.abspath(self.model_path), 'amount_model')
        if os.path.exists(path):
            shutil.rmtree(path)
        if self.amount_prediction_method in [self.RANDOM_FOREST, self.LINEAR_REGRESSION]:
            if not os.path.isdir(path):
                os.makedirs(path)
            path = 'file:{}'.format(path)
            amount_model.save(sc=self.sc, path=path)
        else:
            self.save_data_to_file(amount_model, file_name='amount_model', data_type=self.SAVE_TYPE_MODEL)

        # save trend prediction model
        path = os.path.join(os.path.abspath(self.model_path), 'trend_model')
        if os.path.exists(path):
            shutil.rmtree(path)
        if self.trend_prediction_method in [self.RANDOM_FOREST, self.LOGISTIC_REGRESSION, self.NAIVE_BAYES, self.SVM]:
            if not os.path.isdir(path):
                os.makedirs(path)
            path = 'file:{}'.format(path)
            trend_model.save(sc=self.sc, path=path)

    def train_trend_model(self, model, data, i):
        self.logger.info('Start to train the direction model')
        rdd_data = self.sc.parallelize(data)
        if self.trend_prediction_method == self.RANDOM_FOREST:
            model = RandomForest.trainClassifier(rdd_data, numClasses=2, categoricalFeaturesInfo={}, numTrees=40,
                                                 featureSubsetStrategy="auto", impurity='gini', maxDepth=20,
                                                 maxBins=32)
        elif self.trend_prediction_method == self.NAIVE_BAYES:
            model = NaiveBayes.train(rdd_data)

        elif self.trend_prediction_method == self.LOGISTIC_REGRESSION:
            model = LogisticRegressionWithSGD.train(rdd_data, iterations=100000, step=0.001,
                                                    initialWeights=None if model is None else model.weights)

        elif self.trend_prediction_method == self.SVM:
            model = SVMWithSGD.train(rdd_data, iterations=100000, step=0.001,
                                     initialWeights=None if model is None else model.weights)

        return model

    def train_amount_model(self, model, data, i):
        rdd_data = self.sc.parallelize(data)
        self.logger.info('Start to train the amount model')
        if self.amount_prediction_method == self.ARTIFICIAL_NEURAL_NETWORK:
            input_num = self.feature_num
            layers = [input_num, input_num / 3 * 2, input_num / 3, 1]

            neural_network = NeuralNetworkSpark(layers=layers, bias=0)
            model = neural_network.train(rdd_data, method=neural_network.BP, seed=1234, learn_rate=0.0001,
                                         iteration=15, model=model)
        elif self.amount_prediction_method == self.RANDOM_FOREST:
            model = RandomForest.trainRegressor(rdd_data, categoricalFeaturesInfo={}, numTrees=40,
                                                featureSubsetStrategy="auto", impurity='variance', maxDepth=20,
                                                maxBins=32)

        elif self.amount_prediction_method == self.LINEAR_REGRESSION:
            model = LinearRegressionWithSGD.train(rdd_data, iterations=100000, step=0.001,
                                                  initialWeights=model.weights if model is not None else None)

        else:
            self.logger.error("Unknown training method {}".format(self.amount_prediction_method))
            raise ValueError("Unknown training method {}".format(self.amount_prediction_method))
        return model

    def initialize_model(self):
        if self.trend_prediction_method == self.ARTIFICIAL_NEURAL_NETWORK:
            trend_model = NeuralNetworkModel(
                layers=[self.feature_num, self.feature_num / 3 * 2, self.feature_num / 3, 1])
        else:
            trend_model = None

        amount_model = None

        return trend_model, amount_model

    def evaluate_model(self, trend_model, amount_model, test_features, tomorrow_today):
        predict = self.model_predict(trend_model=trend_model, amount_model=amount_model, test_features=test_features,
                                     tomorrow_today=tomorrow_today)
        predict_rdd = self.sc.parallelize(predict)
        mse = get_MSE(predict_rdd)
        cdc = get_CDC_combine(predict_rdd)
        mape = get_MAPE(predict_rdd)
        mad = get_MAD(predict_rdd)
        if cdc > self.trend_model['cdc']:
            self.trend_model['model'] = trend_model
            self.trend_model['cdc'] = cdc

        if mse < self.amount_model['rmse']:
            self.amount_model['rmse'] = mse
            self.amount_model['model'] = amount_model
        return mse, mape, cdc, mad

    def model_predict(self, trend_model, amount_model, test_features, tomorrow_today):
        amount_type = self.amount_type
        data_parser = self.data_parser
        test_features_rdd = self.sc.parallelize(test_features).cache()
        tomorrow_today_bc = self.sc.broadcast(tomorrow_today)

        test_features_label = test_features_rdd.map(lambda t: t[1])
        if self.amount_prediction_method == self.RANDOM_FOREST:
            amount_predict = amount_model.predict(test_features_rdd.map(lambda t: t[0])).map(
                data_parser.inverse_transform_label).zip(test_features_label)
        else:
            amount_predict = test_features_rdd.map(lambda t: (data_parser.inverse_transform_label(
                amount_model.predict(t[0])), t[1]))

        if self.trend_prediction_method == self.RANDOM_FOREST:
            trend_predict = trend_model.predict(test_features_rdd.map(lambda t: t[0]))
        else:
            trend_predict = test_features_rdd.map(lambda t: trend_model.predict(t[0]))

        predict = amount_predict.zip(trend_predict) \
            .map(lambda t: (tomorrow_today_bc.value[t[0][1]][0], get_predict_result_from_data \
            (t[0][0], t[1], amount_type, tomorrow_today_bc.value[t[0][1]][1]))) \
            .collect()

        # predict = test_features_rdd.map(lambda t: (tomorrow_today_bc.value[t[1]], trend_model.predict(t[0]),
        #                                            data_parser.inverse_transform_label(amount_model.predict(t[0])))) \
        #     .map(lambda t: (t[0][0], get_predict_result_from_data(t[2], t[1], amount_type, t[0][1]))).collect()
        tomorrow_today_bc.unpersist()
        return predict

    def split_data(self, trend_train, amount_train, features):
        training_data_num = len(trend_train)
        train_num = int(0.9 * training_data_num)

        trend_train_bc = self.sc.broadcast(trend_train)
        choice_index = np.random.choice(range(training_data_num), size=train_num, replace=False)
        choice_index_bc = self.sc.broadcast(choice_index)
        # if self.trend_prediction_method in [self.RANDOM_FOREST, self.NAIVE_BAYES]:
        #     trend_train_rdd = self.sc.parallelize([0]).flatMap(lambda x: choice_index_bc.value).map(
        #         lambda x: trend_train_bc.value[x]).collect()
        # else:
        trend_train_rdd = self.sc.parallelize([0]).flatMap(lambda x: trend_train_bc.value).collect()

        amount_train_bc = self.sc.broadcast(amount_train)
        # if self.amount_prediction_method != self.RANDOM_FOREST:
        #     # f = open('test', 'w')
        #     # amount_train_bc.dump(amount_train_bc.value, f)
        #     # f.close()
        #     amount_train_rdd = self.sc.parallelize([0]).flatMap(lambda x: choice_index_bc.value).map(
        #         lambda x: amount_train_bc.value[x]).collect()
        # else:
        amount_train_rdd = self.sc.parallelize([0]).flatMap(lambda x: amount_train_bc.value).collect()
        test_features = self.sc.parallelize([0]).flatMap(lambda x: amount_train_bc.value).map(lambda p: p.features) \
            .zipWithIndex().filter(lambda x: x[1] not in choice_index_bc.value).collect()
        choice_index_bc.unpersist()
        amount_train_bc.unpersist()
        trend_train_bc.unpersist()
        return trend_train_rdd, amount_train_rdd, test_features

    def prepare_data(self, start_date, end_date, test_start_date):

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
        if test_start_date is not None:
            self.train_data_number = 0
            for date_str in self.date_list:
                if string_to_datetime(date_str) < string_to_datetime(test_start_date):
                    self.train_data_number += 1

        self.total_data_num = len(raw_data)

        return raw_data

    def processing_data(self, input_data, train_num):
        if self.data_parser is None:
            self.data_parser = DataParser(label_data_type=self.amount_type)
            train, test, tt = self.data_parser.split_train_test_data(train_num, input_data, True)
        else:
            train, test, tt = self.data_parser.split_train_test_data(train_num, input_data, False)

        self.feature_num = len(train[0][0].features)

        self.train_data_number = train_num
        index = range(self.train_data_number, self.total_data_num)
        test_features = zip(map(lambda p: p.features, test[0]), index)
        return train[0], train[1], test_features, tt

    def predict_historical_data(self, start_date, end_date, test_start_date=None, iterations=10):
        """ Get raw data -> process data -> pca -> normalization -> train -> test """
        self.logger.info('Start to predict stock symbol {}'.format(self.stock_symbol))
        self.logger.info("The amount training method is {}".format(self.amount_prediction_method))
        self.logger.info("The amount type is {}".format(self.amount_type))
        self.logger.info("The direction training method is {}".format(self.trend_prediction_method))

        if self.using_exist_model:
            trend_model, amount_model = self.load_parameters()
        else:
            trend_model = None
            amount_model = None

        # Generate training data
        data_list = self.prepare_data(start_date=start_date, end_date=end_date, test_start_date=test_start_date)
        trend_train, amount_train, all_features, tomorrow_today = self.processing_data(data_list,
                                                                                       self.train_data_number)

        self.logger.info("Initialize Model")
        if not self.using_exist_model:
            trend_model, amount_model = self.initialize_model()

        self.logger.info('Start to training model')
        for i in range(iterations):
            self.logger.info("Epoch {} starts".format(i))
            trend_train_x, amount_train_x, test_features_x = self.split_data(
                trend_train=trend_train, amount_train=amount_train, features=all_features)
            trend_model = self.train_trend_model(data=trend_train_x, model=trend_model, i=i)
            amount_model = self.train_amount_model(model=amount_model, data=amount_train_x, i=i)

            self.logger.info("Epoch {} finishes".format(i))
            mse, mape, cdc, mad = self.evaluate_model(trend_model=trend_model, amount_model=amount_model,
                                                      test_features=test_features_x, tomorrow_today=tomorrow_today)
            self.logger.info("Current MSE is {:.4f}".format(mse))
            self.logger.info("Current MAD is {:.4f}".format(mad))
            self.logger.info("Current MAPE is {:.4f}%".format(mape))
            self.logger.info("Current CDC is {:.4f}%".format(cdc))

        # if train ratio is at that level, means that target want the model file, not the
        if test_start_date is None:
            return self.trend_model['model'], self.amount_model['model']

        # Data prediction part
        self.logger.info("Start to use the model to predict price")
        predict = self.model_predict(trend_model=self.trend_model['model'], amount_model=self.amount_model['model'],
                                     test_features=all_features, tomorrow_today=tomorrow_today)

        self.save_data_to_file(predict, "predict_result.csv", self.SAVE_TYPE_OUTPUT)
        self.save_model(trend_model=trend_model, amount_model=amount_model)

        return predict
