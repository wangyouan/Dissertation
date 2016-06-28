#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: inference_system
# Author: Mark Wang
# Date: 1/6/2016

import os
import sys
import shutil
import datetime

from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LinearRegressionWithSGD, LabeledPoint
from pandas.tseries.offsets import CustomBusinessDay

from StockInference.constant import Constants
from StockInference.DataCollection.data_collect import DataCollect
from StockInference.Regression.distributed_neural_network import NeuralNetworkSpark, NeuralNetworkModel
from StockInference.util.data_parse import min_max_de_normalize, get_MSE, get_MAD, get_MAPE, get_CDC
from StockInference.DataParser.data_parser import DataParser
from StockInference.util.date_parser import get_ahead_date, string_to_datetime
from StockInference.util.file_operation import load_data_from_file, save_data_to_file
from StockInference.util.hongkong_calendar import HongKongCalendar

interest_rate_path = "interest_rate"

if sys.platform == 'darwin':
    interest_rate_path = os.path.join('..', interest_rate_path)


class InferenceSystem(Constants):
    def __init__(self, stock_symbol, data_folder_path=None, training_method=None, features=None, output_file_path=None,
                 model_path=None, using_exist_model=False):
        self.stock_symbol = stock_symbol
        conf = SparkConf()
        conf.setAppName("{}_{}".format(self.__class__.__name__, stock_symbol))
        self.sc = SparkContext.getOrCreate(conf=conf)
        self.train_data = None
        self.test_data = None
        self.test_data_features = None
        self.total_data_num = 0
        self.data_parser = None
        self.date_list = []
        self.data_path = data_folder_path
        self.output_path = output_file_path
        self.data_features = features
        self.model_path = model_path
        self.feature_num = None
        self.using_exist_model = using_exist_model
        if training_method is None:
            self.training_method = self.ARTIFICIAL_NEURAL_NETWORK
        else:
            self.training_method = training_method
        log4jLogger = self.sc._jvm.org.apache.log4j
        self.logger = log4jLogger.LogManager.getLogger(self.__class__.__name__)
        self.predict_model = {'model': None, 'mse': float('inf')}

    def get_train_test_data(self, test_start_date, start_date, end_date):

        self.logger.info('Get train and testing data')
        self.logger.info('Training / Testing ratio is {}'.format(test_start_date))
        self.logger.info('Start date is {}, end date is {}'.format(start_date, end_date))

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
                                      data_file_path=self.data_path, logger=self.sc._jvm.org.apache.log4j.LogManager)
        data_collection.set_interest_rate_path(interest_rate_path)
        if self.data_features is None:
            self.data_features = required_info

        # features = {self.FUNDAMENTAL_ANALYSIS: [self.ONE_YEAR]}
        self.logger.info("No previous data, will collected them from Internet")
        raw_data = data_collection.get_raw_data(label_info=self.data_features[self.PRICE_TYPE],
                                                required_info=self.data_features)
        self.date_list = data_collection.get_date_list()
        train_num = 0
        if test_start_date is not None:
            for date_str in self.date_list:
                if string_to_datetime(date_str) < string_to_datetime(test_start_date):
                    train_num += 1

        # debug
        # raw_data_file = open(os.path.join('../output', "raw.dat"), 'w')
        # import pickle
        # pickle.dump(raw_data, raw_data_file)
        # raw_data_file.close()
        #
        # f = open('text.csv', 'w')
        # f.write(
        #     'date,open,high,low,close,macd1,macd2,sma_3,sma_13,sma_21,ema_5,ema_13,ema_21,roc_13,roc_21,rsi_9,rsi_14,rsi_21,us10y,us30y,fxi,hsi,usdhkd,eurhkd,oneyear,halfyear,overnight,golden_price,tomm_close\n')
        # date_list = data_collection.get_date_list()
        # for i in range(len(raw_data)):
        #     f.write("{},{},{}\n".format(date_list[i], ','.join(map(str, raw_data[i].features)), raw_data[i].label))
        # f.close()
        # raise ValueError("Warn SB")

        # Split train and test
        if self.data_parser is None:
            n_components = None
            self.data_parser = DataParser(n_components=n_components)

            self.train_data, self.test_data, self.test_data_features = self.data_parser.split_train_test_data(
                train_num=train_num, raw_data=raw_data, fit_transform=True)
        else:
            self.train_data, self.test_data, self.test_data_features = self.data_parser.split_train_test_data(
                train_num=train_num, raw_data=raw_data, fit_transform=False)
        self.total_data_num = len(raw_data)
        self.feature_num = len(self.train_data[0].features)
        self.logger.info('Get train and testing data finished')
        self.logger.info("Total data num is {}, train data num is {}".format(self.total_data_num, len(self.train_data)))
        self.logger.info("Input feature number is {}".format(self.feature_num))

        train_num = len(self.train_data)
        train_features = []
        for i in range(train_num):
            train_features.append(raw_data[i].features[:4])

        return train_features

    def load_parameters(self):
        if self.using_exist_model and self.model_path:
            self.training_method = self.load_data_from_file(self.SAVE_TYPE_MODEL, 'train_method')
            self.data_features = self.load_data_from_file(self.SAVE_TYPE_MODEL, 'features')
            self.data_parser = self.load_data_from_file(self.SAVE_TYPE_MODEL, 'data_parser')
            if self.training_method == self.RANDOM_FOREST:
                model = RandomForestModel.load(sc=self.sc, path='file:{}'.format(
                    os.path.join(os.path.abspath(self.model_path), 'model')))
            else:
                model = self.load_data_from_file(self.SAVE_TYPE_MODEL, 'model')
            return model
        else:
            return None

    def save_parameters(self, model):
        self.save_data_to_file(self.training_method, 'train_method', self.SAVE_TYPE_MODEL)
        self.save_data_to_file(self.data_features, 'features', self.SAVE_TYPE_MODEL)
        self.save_data_to_file(self.data_parser, 'data_parser', self.SAVE_TYPE_MODEL)
        if self.training_method == self.RANDOM_FOREST:
            path = os.path.join(os.path.abspath(self.model_path), 'model')
            if os.path.exists(path):
                shutil.rmtree(path)

            if not os.path.isdir(path):
                os.makedirs(path)
            path = 'file:{}'.format(path)
            model.save(sc=self.sc, path=path)
        else:
            self.save_data_to_file(model, 'model', self.SAVE_TYPE_MODEL)

    def save_data_to_file(self, data, file_name, data_type):
        if not file_name.endswith('dat') and data_type != self.SAVE_TYPE_OUTPUT:
            file_name = '%s.dat' % file_name

        if data_type == self.SAVE_TYPE_MODEL:
            path = self.model_path
        elif data_type == self.SAVE_TYPE_INPUT:
            path = self.data_path
        elif data_type == self.SAVE_TYPE_OUTPUT:
            path = self.output_path

        else:
            self.logger.info("Unknown data type, cannot save")
            path = None

        if path is None:
            self.logger.warn("Cannot save data without set target path")
            return
        elif not os.path.isdir(path):
            os.makedirs(path)

        if data_type == self.SAVE_TYPE_OUTPUT:
            if hasattr(self, 'train_data_number'):
                train_data_num = self.train_data_number
            else:
                train_data_num = len(self.train_data)
            test_date_list = self.date_list[train_data_num:]
            output_file = open(os.path.join(path, file_name), "w")
            output_file.write("date,origin,predict\n")
            test_date_list = test_date_list[1:]
            test_date_list.append(get_ahead_date(test_date_list[-1], -1))
            for i in range(self.total_data_num - train_data_num):
                output_file.write("%s,%2f,%2f\n" % (
                    test_date_list[i], data[i][0], data[i][1]))
            output_file.close()

        else:
            file_path = os.path.join(path, file_name)
            save_data_to_file(file_path, data)

    def load_data_from_file(self, data_type, file_name):
        if not file_name.endswith('dat'):
            file_name = '%s.dat' % file_name
        if data_type == self.SAVE_TYPE_OUTPUT:
            path = os.path.join(self.output_path, file_name)
            if os.path.isfile(path):
                return load_data_from_file(path)
        elif data_type == self.SAVE_TYPE_INPUT:
            path = os.path.join(self.data_path, file_name)
            if os.path.isfile(path):
                return load_data_from_file(path)
        elif data_type == self.SAVE_TYPE_MODEL:
            path = os.path.join(self.model_path, file_name)
            if os.path.isfile(path):
                return load_data_from_file(path)

    def model_training(self, training_data, model):
        if self.training_method == self.ARTIFICIAL_NEURAL_NETWORK:
            input_num = len(self.train_data[0].features)
            layers = [input_num, input_num / 3 * 2, input_num / 3, 1]

            neural_network = NeuralNetworkSpark(layers=layers, bias=0)
            model = neural_network.train(training_data, method=neural_network.BP, seed=1234, learn_rate=0.0001,
                                         iteration=10, model=model)
        elif self.training_method == self.RANDOM_FOREST:

            model = RandomForest.trainRegressor(training_data, categoricalFeaturesInfo={}, numTrees=40,
                                                featureSubsetStrategy="auto", impurity='variance', maxDepth=20,
                                                maxBins=32, seed=1234)

        elif self.training_method == self.LINEAR_REGRESSION:
            model = LinearRegressionWithSGD.train(training_data, iterations=10000, step=0.001,
                                                  initialWeights=model.weights if model is not None else None)

        else:
            self.logger.error("Unknown training method {}".format(self.training_method))
            raise ValueError("Unknown training method {}".format(self.training_method))
        return model

    def model_prediction(self, model, testing_data, testing_data_features):

        if self.training_method != self.RANDOM_FOREST:

            # predicting
            predict = testing_data.map(lambda p: (p.label, model.predict(p.features))) \
                .zip(testing_data_features) \
                .map(lambda (p, v): (p[0], min_max_de_normalize(p[1], v))).cache()
        else:
            predict = model.predict(testing_data.map(lambda x: x.features))
            predict = testing_data.zip(predict).zip(testing_data_features) \
                .map(lambda (m, n): (m[0].label, min_max_de_normalize(m[1], n))).cache()
        return predict

    def randomly_split_data(self, total_data, ratio=0.9):
        train, test = total_data.randomSplit([ratio, 1 - ratio])
        train = train.map(lambda p: p[0])
        test_features = test.map(lambda p: p[1])
        test = test.map(lambda p: p[0])
        return train, test, test_features

    def initialize_model(self):
        if self.training_method == self.ARTIFICIAL_NEURAL_NETWORK:
            model = NeuralNetworkModel(layers=[self.feature_num, self.feature_num / 3 * 2, self.feature_num / 3, 1])
        else:
            model = None

        return model

    def evaluate_model_performance(self, model, test_data, test_features):
        test_data = test_data.zip(test_features).map(
            lambda (d, f): LabeledPoint(label=min_max_de_normalize(d.label, f), features=d.features))
        predict = self.model_prediction(model, testing_data=test_data, testing_data_features=test_features)
        mse = get_MSE(predict)
        mape = get_MAPE(predict)
        cdc = get_CDC(predict)
        mad = get_MAD(predict)
        if mse < self.predict_model['mse']:
            self.predict_model['model'] = model
            self.predict_model['mse'] = mse

        return mse, mape, cdc, mad

    def predict_historical_data(self, test_start_date, start_date, end_date, iterations=10):

        """ Get raw data -> process data -> pca -> normalization -> train -> test """
        self.logger.info('Start to predict stock symbol {}'.format(self.stock_symbol))
        self.logger.info("The training method is {}".format(self.training_method))

        if self.using_exist_model:
            model = self.load_parameters()
        else:
            model = None

        # Generate training data
        train_features = self.get_train_test_data(test_start_date, start_date=start_date, end_date=end_date)

        training_data = self.sc.parallelize(zip(self.train_data, train_features)).cache()

        self.logger.info("Initialize Model")
        if not self.using_exist_model:
            model = self.initialize_model()

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

        # if train ratio is at that level, means that target want the model file, not the
        if test_start_date is None:
            return self.predict_model['model']

        model = self.predict_model['model']

        # Data prediction part
        self.logger.info("Start to use the model to predict price")
        testing_data = self.sc.parallelize(self.test_data)
        testing_data_features = self.sc.parallelize(self.test_data_features)
        predict = self.model_prediction(model, testing_data=testing_data, testing_data_features=testing_data_features)

        self.save_data_to_file(predict.collect(), "predict_result.csv", self.SAVE_TYPE_OUTPUT)
        self.save_parameters(model)

        return predict

    def get_future_stock_price(self, start_history=None):
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
        if self.model_path is not None and self.using_exist_model:
            model = self.load_parameters()
        else:

            if start_history is None:
                start_date = datetime.datetime(end_day.year - 2, end_day.month, end_day.day)
                start_date += cday
            else:
                start_date = start_history

            if not isinstance(start_date, str):
                start_date = start_date.strftime("%Y-%m-%d")

            model = self.predict_historical_data(1, start_date=start_date, end_date=end_date)
        data_collection = DataCollect(self.stock_symbol, end_date, end_date, data_file_path=self.data_path,
                                      logger=self.sc._jvm.org.apache.log4j.LogManager)
        data_collection.set_interest_rate_path(interest_rate_path)
        data = data_collection.get_raw_data(self.data_features[self.PRICE_TYPE], required_info=self.data_features)
        predict_features = self.data_parser.transform(data)[0]
        predict_price = model.predict(predict_features)

        return predict_date, min_max_de_normalize(predict_price, features=data[0])
