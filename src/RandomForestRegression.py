#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: RandomForestRegression
# Author: Mark Wang
# Date: 20/4/2016

from pyspark.mllib.tree import RandomForest, RandomForestModel

from constant import *
from parse_data import DataParser
from __init__ import sc, sql_context


def price_predict(path, windows=5):
    input_data = DataParser(path=path, window_size=windows)
    open_rows, close_rows = input_data.get_n_days_history_data(data_type=LABEL_POINT, spark_context=sc)
    train_data_len = int(0.8 * len(open_rows))

    #handle open data
    open_train_rows = open_rows[:train_data_len]
    open_test_rows = open_rows[(train_data_len - 1):]
    open_train_df = DataParser.convert_to_data_frame(open_train_rows, sc=sc, sql=sql_context)
    open_test_df = DataParser.convert_to_data_frame(open_test_rows, sc=sc, sql=sql_context)
    open_model = RandomForest.trainRegressor(open_train_df, categoricalFeaturesInfo={},
                                             numTrees=3, featureSubsetStrategy="auto",
                                             impurity='variance', maxDepth=4, maxBins=32)
    open_prediction = open_model.predict(open_test_df.map(lambda x: x.features))
    print("Precision:" + str(evaluator.evaluate(open_prediction_labels)))

    close_train_rows = close_rows[:train_data_len]
    close_test_rows = close_rows[(train_data_len - 1):]
    close_train_df = DataParser.convert_to_data_frame(close_train_rows, sc=sc, sql=sql_context)
    close_test_df = DataParser.convert_to_data_frame(close_test_rows, sc=sc, sql=sql_context)
    close_trainer = MultilayerPerceptronClassifier(maxIter=100, layers=[4, 5, 4, 3], blockSize=128,
                                                   featuresCol=FEATURES, labelCol=LABEL, seed=1234)
    close_model = close_trainer.fit(close_train_df)
    close_result = close_model.transform(close_test_df)
    close_prediction_labels = close_result.select(PREDICTION, LABEL)
    print("Precision:" + str(evaluator.evaluate(close_prediction_labels)))