#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: regression_method
# Author: Mark Wang
# Date: 14/10/2016


from keras.models import Sequential
from keras.layers.core import Dense

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from elephas import optimizers as elephas_optimizers


def ann_train(spark, features, labels, nb_epoch=10, batch_size=64):
    rdd = to_simple_rdd(spark.sparkContext, features=features, labels=labels)

    model = Sequential()
    model.add(Dense(18, input_dim=26, activation='sigmoid'))
    model.add(Dense(12, activation='sigmoid'))
    model.add(Dense(1))

    spark_model = SparkModel(sc=spark.sparkContext,
                             master_network=model,
                             optimizer=elephas_optimizers.adam(),
                             mode='asynchronous',
                             master_loss='mse')

    spark_model.train(rdd, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2, validation_split=0.1)
    return spark_model
