#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: test_elephas
# Author: Mark Wang
# Date: 13/10/2016

from pyspark.sql import SparkSession

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from elephas import optimizers as elephas_optimizers

from sklearn.preprocessing import MinMaxScaler

import pandas as pd

batch_size = 64
nb_epoch = 10

train = pd.read_pickle('train_df.p')
test = pd.read_pickle('test_df.p')
test = test.dropna(how='any')

transformer = MinMaxScaler()

y_train = transformer.fit_transform(train['Target'].values.reshape(-1, 1))
del train['Target']
y_test = transformer.transform(test['Target'].values.reshape(-1, 1))
del test['Target']

model = Sequential()
model.add(Dense(18, input_dim=26))
model.add(Activation('sigmoid'))
model.add(Dense(6))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

spark = SparkSession.builder.appName('ElephasTest').getOrCreate()
rdd = to_simple_rdd(spark.sparkContext, train, y_train)

sgd = SGD(lr=0.1)
adagrad = elephas_optimizers.Adagrad()
spark_model = SparkModel(spark.sparkContext,
                         model,
                         optimizer=adagrad,
                         frequency='epoch',
                         mode='asynchronous',
                         master_loss='mse',
                         num_workers=2, master_optimizer=sgd)

# Train Spark model
spark_model.train(rdd, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2, validation_split=0.1)

# Evaluate Spark model by evaluating the underlying model
score = spark_model.master_network.evaluate(test.values, y_test, verbose=2)
print('Test accuracy:', score[1])
print spark_model.predict(test.values)
print y_test
