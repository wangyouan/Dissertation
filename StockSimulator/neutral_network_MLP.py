#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: neutral_network_MLP
# Author: Mark Wang
# Date: 17/4/2016

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from parse_data import DataParser
from constant import *
from __init__ import load_spark_context


def price_predict(path, windows=5, spark_contest=None, sql_context=None):
    if spark_contest is None:
        spark_contest, sql_context = load_spark_context()
    input_data = DataParser(path=path, window_size=windows)
    close_train_df, close_test_df, open_train_df, open_test_df = input_data.get_n_days_history_data(
        data_type=DATA_FRAME, spark_context=spark_contest, sql_context=sql_context)
    evaluator = MulticlassClassificationEvaluator(metricName=PREDICTION)

    # handle open data
    open_trainer = MultilayerPerceptronClassifier(maxIter=1, layers=[4, 5, 4, 3], blockSize=128,
                                                  featuresCol=FEATURES, labelCol=LABEL, seed=1234)
    open_model = open_trainer.fit(open_train_df)
    open_result = open_model.transform(open_test_df)
    open_prediction_labels = open_result.select(PREDICTION, LABEL)
    print("Precision:" + str(evaluator.evaluate(open_prediction_labels)))

    # handle close data
    close_trainer = MultilayerPerceptronClassifier(maxIter=100, layers=[4, 5, 4, 3], blockSize=128,
                                                   featuresCol=FEATURES, labelCol=LABEL, seed=1234)
    close_model = close_trainer.fit(close_train_df)
    close_result = close_model.transform(close_test_df)
    close_prediction_labels = close_result.select(PREDICTION, LABEL)
    print("Precision:" + str(evaluator.evaluate(close_prediction_labels)))


if __name__ == "__main__":
    # try:
    price_predict(r'../data/0001.HK.csv')
    # finally:
        # sc.stop()
