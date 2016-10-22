#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: neural_network
# Author: Mark Wang
# Date: 15/10/2016

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from elephas import optimizers as elephas_optimizers


class KerasNeuralNetworkSpark(object):
    def __init__(self, layers, spark, batch_size=64, epoch=10, num_workers=2, predictionCol='prediction',
                 labelCol='target', featuresCol='feature'):
        self._batch_size = batch_size
        self._epoch = epoch
        self._model = None
        self._spark = spark
        self._labels = labelCol
        self._features = featuresCol
        self._prediction = predictionCol
        self._layers = layers
        self._worker_num = num_workers
        self._build_model()

    def _build_model(self):
        model = Sequential()
        adam = elephas_optimizers.Adam()
        layers = self._layers
        model.add(Dense(layers[1], input_dim=layers[0], init='normal', activation='relu'))
        for i in range(2, len(layers) - 1):
            model.add(Dense(layers[i], activation='relu'))

        model.add(Dense(layers[-1], activation='sigmoid'))
        self._model = SparkModel(self._spark.sparkContext, model,
                                 optimizer=adam,
                                 frequency='epoch',
                                 mode='asynchronous',
                                 master_loss='mse',
                                 num_workers=self._worker_num)

    def fit(self, df):
        if hasattr(self._model, 'server'):
            self._model.server.terminate()
        pdf = df.toPandas()

        rdd = to_simple_rdd(self._spark.sparkContext, pdf[self._features], pdf[self._labels])
        self._model.train(rdd, self._epoch, self._batch_size, 0, 0.1)

    def transform(self, df):
        pdf = df.toPandas()
        # df.write.save('test_df.parquet')
        pnparray = pdf[self._features].values
        container = np.zeros((pnparray.shape[0], len(pnparray[0])))
        for i in range(pnparray.shape[0]):
            container[i, :] = pnparray[i][:]
        result = self._model.predict(container)

        pdf[self._prediction] = result

        # import pickle
        # with open('ann_result.p', 'w') as f:
        #     pickle.dump(result, f)

        # result_df = pd.DataFrame(pdf
        new_df = self._spark.createDataFrame(pdf)
        # df.join(new_df)
        return new_df

    def stop_server(self):
        if hasattr(self._model, 'server') and hasattr(self._model.server, 'terminate'):
            self._model.server.terminate()
