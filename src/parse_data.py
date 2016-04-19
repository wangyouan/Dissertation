#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: parse_data
# Author: Mark Wang
# Date: 17/4/2016

from pyspark import SparkContext
from pyspark.sql import Row, SQLContext
from pyspark.sql.types import *
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector, VectorUDT

from constant import *


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


    def normalize_data(self, close_data=None, open_data=None, features=None):
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


if __name__ == "__main__":
    from pyspark.ml.classification import MultilayerPerceptronClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    import pprint

    test = DataParser(r"../data/0001.HK.csv", 5)
    data_list = test.load_data_from_yahoo_csv()
    data_list = data_list[:100]
    sc = SparkContext(appName="DataParserTest")
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.OFF)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.OFF)

    sql_context = SQLContext(sc)
    try:
        a, b = test.get_n_days_history_data(data_list=data_list, sql_context=sql_context, spark_context=sc)
        # print data_list
        # print a.select('label').collect()
        # print b.select('label').collect()
        f = open("close_collect.txt", "w")
        f.write(pprint.pformat(a.collect(), width=120))
        f.close()
        trainer = MultilayerPerceptronClassifier(maxIter=100, layers=[4, 5, 5], blockSize=128,
                                                 featuresCol=FEATURES, labelCol=LABEL, seed=1234)
        trainer.fit(a)
    finally:
        sc.stop()
