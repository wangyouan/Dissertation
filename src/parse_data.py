#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: parse_data
# Author: Mark Wang
# Date: 17/4/2016

from constant import *
from pyspark.mllib.linalg import SparseVector, VectorUDT
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.types import *


def avg(data_list):
    """
    Use to calculate simple moving average
    :param data_list: the list need to be calculated
    :return: the moving average of the data list
    """
    return sum(data_list) / float(len(data_list))


class DataParser(object):
    """
    Use to handle data

    self.__path: CRV
    """

    def __init__(self, path, window_size=1):
        self.__path = path
        self.__days = window_size
        self.features = None

    def get_n_days_history_data(self, data_list=None, data_type="DataFrame", n_days=None, sql_context=None,
                                spark_context=None, train_ratio=0.8, normalized=True):
        """
        Use to handle yahoo finance history data, return will be DataFrame of RDD
        :param n_days: get how many days
        :param data_type: "DataFrame" or "RDD"
        :param data_list: which part of data is needed [Open,High,Low,Close]
        :return: Required two required data, label normalized close price, features: [highest price, lowest price,
                 average close price]
        """
        if data_list is None:
            data_list = self.load_data_from_yahoo_csv()

        close_data, open_data, self.features = self.get_time_series_data(data_list=data_list, window_size=n_days)
        train_len = int(train_ratio * len(open_data))
        close_train_list = []
        open_train_list = []
        close_test_list = []
        open_test_list = []

        def normalize(price, max_price, min_price):
            if not normalized:
                return price
            if max_price - min_price < 1e-4:
                return 0
            return (2 * price - (max_price + min_price)) / (max_price - min_price)

        if data_type == DATA_FRAME:

            for i in range(train_len):
                # feature = SparseVector(4, [(k, j) for k, j in enumerate(self.features[i])])
                # close_train_list.append((close_normalized[i], feature))
                # open_train_list.append((open_normalized[i], feature))

                close_train_list.append((normalize(close_data[i], self.features[i][1], self.features[i][2]),
                                         Vectors.dense(self.features[i])))
                open_train_list.append((normalize(open_data[i], self.features[i][1], self.features[i][2]),
                                        Vectors.dense(self.features[i])))

            for i in range(train_len - 1, len(self.features)):
                # feature = SparseVector(4, [(k, j) for k, j in enumerate(self.features[i])])
                # close_test_list.append((close_normalized[i], feature))
                # open_test_list.append((open_normalized[i], feature))

                close_test_list.append((close_data[i], Vectors.dense(self.features[i])))
                open_test_list.append((open_data[i], Vectors.dense(self.features[i])))

            if sql_context is not None and spark_context is not None:
                close_train_list = self.convert_to_data_frame(input_rows=close_train_list, sql=sql_context,
                                                              sc=spark_context)
                open_train_list = self.convert_to_data_frame(input_rows=open_train_list, sql=sql_context,
                                                             sc=spark_context)
                close_test_list = self.convert_to_data_frame(input_rows=close_test_list, sql=sql_context,
                                                             sc=spark_context)
                open_test_list = self.convert_to_data_frame(input_rows=open_test_list, sql=sql_context,
                                                            sc=spark_context)
        elif data_type == LABEL_POINT:
            for i in range(train_len):
                close_train_list.append(LabeledPoint(features=self.features[i],
                                                     label=normalize(close_data[i], self.features[i][1],
                                                                     self.features[i][2])))
                open_train_list.append(LabeledPoint(features=self.features[i],
                                                    label=normalize(close_data[i], self.features[i][1],
                                                                    self.features[i][2])))

            for i in range(train_len - 1, len(self.features)):
                close_test_list.append(LabeledPoint(features=self.features[i], label=close_data[i]))
                open_test_list.append(LabeledPoint(features=self.features[i], label=open_data[i]))

            if spark_context is not None:
                close_train_list = spark_context.parallelize(close_train_list)
                open_train_list = spark_context.parallelize(open_train_list)
                close_test_list = spark_context.parallelize(close_test_list)
                open_test_list = spark_context.parallelize(open_test_list)

        return close_train_list, close_test_list, open_train_list, open_test_list

    def get_n_days_history_data_old(self, data_list=None, data_type="DataFrame", n_days=None, sql_context=None,
                                    spark_context=None):
        """
        Use to handle yahoo finance history data, return will be DataFrame of RDD
        :param n_days: get how many days
        :param data_type: "DataFrame" or "RDD"
        :param data_list: which part of data is needed [Open,High,Low,Close]
        :return: Required two required data, label normalized close price, features: [highest price, lowest price,
                 average close price]
        """
        if data_list is None:
            data_list = self.load_data_from_yahoo_csv()

        close_data, open_data, self.features = self.get_time_series_data(data_list=data_list, window_size=n_days)
        open_normalized, close_normalized = self.normalize_data(close_data=close_data, open_data=open_data,
                                                                features=self.features)
        if data_type == DATA_FRAME:
            # PriceData = Row("features", "labels")
            # FeatureData = Row(OPEN, HIGH, LOW, CLOSE)
            close_row = []
            open_row = []

            for i in range(len(self.features)):
                feature = SparseVector(4, [(k, j) for k, j in enumerate(self.features[i])])
                close_row.append((close_normalized[i], feature))
                open_row.append((open_normalized[i], feature))

            if sql_context is None or spark_context is None:
                return close_row, open_row
            else:
                close_data_frame = self.convert_to_data_frame(input_rows=close_row, sql=sql_context,
                                                              sc=spark_context)
                open_data_frame = self.convert_to_data_frame(input_rows=open_row, sql=sql_context,
                                                             sc=spark_context)
                return close_data_frame, open_data_frame
        elif data_type == LABEL_POINT:
            close_labels = []
            open_labels = []
            for i in range(len(self.features)):
                close_labels.append(LabeledPoint(features=self.features[i], label=close_normalized[i]))
                open_labels.append(LabeledPoint(features=self.features[i], label=open_normalized[i]))

            if spark_context is None:
                return close_labels, open_labels
            else:

                close_data = spark_context.parallelize(close_labels)
                open_data = spark_context.parallelize(open_labels)
                return close_data, open_data

    @staticmethod
    def convert_to_data_frame(input_rows, sql, sc):
        schema = StructType([StructField(LABEL, DoubleType(), True),
                             StructField(FEATURES, VectorUDT(), True)])
        input_rdd = sc.parallelize(input_rows)
        return input_rdd.toDF(schema)

    def normalize_data(self, close_data=None, open_data=None, features=None, normalized=True):
        if not normalized:
            return close_data, open_data

        if features is None:
            features = self.features

        data_len = len(features)

        def normalize(price, max_price, min_price):
            if max_price - min_price < 1e-4:
                return 0
            return (2 * price - (max_price + min_price)) / (max_price - min_price)

        close_normalize_data = []
        open_normalize_data = []
        for i in range(data_len):
            if close_data:
                close_price = normalize(close_data[i], features[i][1], features[i][2])
                close_normalize_data.append(close_price)

            if open_data:
                open_price = normalize(open_data[i], features[i][1], features[i][2])
                open_normalize_data.append(open_price)

        return open_normalize_data, close_normalize_data

    def de_normalize_data(self, close_data=None, open_data=None, features=None):
        """
        Use to de-normalize price data
        :param close_data:
        :param open_data:
        :param features:
        :return:
        """

        def de_normalize(price, max_price, min_price):
            return (price * (max_price - min_price)) / 2 + (max_price + min_price) / 2

        if features is None:
            features = self.features

        close_origin_data = []
        open_origin_data = []
        for i in range(len(features)):
            if close_data:
                close_price = de_normalize(close_data[i], features[i][1], features[i][2])
                close_origin_data.append(close_price)

            if open_data:
                open_price = de_normalize(open_data[i], features[i][1], features[i][2])
                open_origin_data.append(open_price)

        return open_origin_data, close_origin_data

    def load_data_from_yahoo_csv(self, path=None):
        """
        Use to load csv file
        :param path: file path
        :return: te data list in csv file
        """
        if path is None:
            path = self.__path

        f = open(path)
        data_str = f.read()
        f.close()
        data_list = [map(float, i.split(',')[1:]) for i in data_str.split('\n')[1:-1]]
        data_list.reverse()
        return data_list

    def get_time_series_data(self, data_list, window_size=None):
        """
        Get time series from given data
        :param data_list: [open, high, low, close]
        :param window_size: the number of days
        :return:
        """
        if window_size is None:
            window_size = self.__days

        data_num = len(data_list)
        close_data = [data_list[i][3] for i in range(window_size, data_num)]
        open_data = [data_list[i][0] for i in range(window_size, data_num)]
        window_data = [data_list[i:(i + window_size)] for i in range(data_num - window_size)]

        # create feature prices
        max_price = []
        min_price = []
        close_avg_price = []
        open_avg_price = []
        for window in window_data:
            max_price.append(max([i[1] for i in window]))
            min_price.append(min([i[2] for i in window]))
            close_avg_price.append(avg([i[3] for i in window]))
            open_avg_price.append(avg([i[0] for i in window]))
        features = zip(open_avg_price, max_price, min_price, close_avg_price)

        return close_data, open_data, features

    @staticmethod
    def de_normalize(p, feature=None):
        if feature is None:
            if isinstance(p, list):
                price = p[0]
                max_price = p[1]
                min_price = p[2]
            else:
                price = p.label
                max_price = p.features[1]
                min_price = p.features[2]
        else:
            price = p
            max_price = feature[1]
            min_price = feature[2]

        return price * (max_price - min_price) / 2 + (max_price + min_price) / 2

    @staticmethod
    def get_MSE(label_and_prediction):
        return label_and_prediction.map(lambda (v, p): (v - p) * (v - p)).sum() / float(label_and_prediction.count())

    @staticmethod
    def get_MAD(label_prediction):
        return label_prediction.map(lambda (v, p): abs(v - p)).sum() / float(label_prediction.count())

    @staticmethod
    def get_MAPE(label_prediction):
        return label_prediction.map(lambda (v, p): abs((v - p) / float(v))).sum() / float(label_prediction.count())
