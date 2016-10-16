#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: train_with_spark
# Author: Mark Wang
# Date: 16/10/2016

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, MultilayerPerceptronClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor

from stockforecaster.constant import Constants
from stockforecaster.regression_method.neural_network_regression_spark import KerasNeuralNetworkSpark

LINEAR_REGRESSION_ITERATION_TIMES = 100000
RANDOM_FOREST_TREE_NUMBER = 30
RANDOM_FOREST_DEPTH = 20
WORKER_NUMBERS = 2


class SparkTrainingSystem(Constants):
    def __init__(self, spark, training_method):
        self._spark = spark
        self._train_method = training_method
        self.logger = spark._jvm.org.apache.log4j.LogManager.getLogger(self.__class__.__name__)
        self._model = None

    def train(self, features, label):
        if isinstance(self._train_method, dict):
            features[self.CHANGE_AMOUNT] = label[self.CHANGE_AMOUNT]
            features[self.CHANGE_DIRECTION] = label[self.CHANGE_DIRECTION]
        else:
            features[self.TARGET_PRICE] = label[self.TARGET_PRICE]
        return self._train_model_spark(data=features)

    def predict(self, features):
        return self._predict(features)

    def _predict(self, features):
        df = self._prepare_data_spark(features)
        if isinstance(self._model, dict):
            df = self._model[self.CHANGE_DIRECTION].transform(df)
            df = self._model[self.CHANGE_AMOUNT].transform(df)
        else:
            df = self._model.transform(df)

        pdf = df.toPandas()
        pdf['Date'] = features.index
        return pdf.set_index('Date')

    def _train_model_spark(self, data):
        df = self._prepare_data_spark(data)
        input_num = len(data.keys().difference({self.CHANGE_AMOUNT, self.CHANGE_DIRECTION, self.TARGET_PRICE,
                                                self.TODAY_PRICE}))
        ann_layers = [input_num, input_num / 3 * 2, input_num / 3, 2]
        if isinstance(self._train_method, dict):
            self._model = {self.CHANGE_AMOUNT: None,
                           self.CHANGE_DIRECTION: None}

            if self._train_method[self.CHANGE_AMOUNT] == self.LINEAR_REGRESSION:
                lr = LinearRegression(featuresCol="features", labelCol=self.CHANGE_AMOUNT,
                                      maxIter=LINEAR_REGRESSION_ITERATION_TIMES,
                                      predictionCol='AmountPrediction')
                self._model[self.CHANGE_AMOUNT] = lr.fit(df)
            elif self._train_method[self.CHANGE_AMOUNT] == self.RANDOM_FOREST:
                rfr = RandomForestRegressor(featuresCol="features", labelCol=self.CHANGE_AMOUNT,
                                            numTrees=RANDOM_FOREST_TREE_NUMBER,
                                            maxDepth=RANDOM_FOREST_DEPTH, predictionCol='AmountPrediction')
                self._model[self.CHANGE_AMOUNT] = rfr.fit(df)
            elif self._train_method[self.CHANGE_AMOUNT] == self.ARTIFICIAL_NEURAL_NETWORK:
                ann_layers[-1] = 1
                self._model[self.CHANGE_AMOUNT] = KerasNeuralNetworkSpark(layers=ann_layers, spark=self._spark,
                                                                          num_workers=WORKER_NUMBERS, epoch=100,
                                                                          featuresCol="features",
                                                                          labelCol=self.CHANGE_AMOUNT,
                                                                          predictionCol='AmountPrediction'
                                                                          )
                self._model[self.CHANGE_AMOUNT].fit(df)
            else:
                self.logger.warn('Unsupported training method {}'.format(self._train_method))
                raise ValueError('Unsupported training method {}'.format(self._train_method))

            if self._train_method[self.CHANGE_DIRECTION] == self.LOGISTIC_REGRESSION:
                lr = LogisticRegression(featuresCol="features", labelCol=self.CHANGE_DIRECTION,
                                        maxIter=LINEAR_REGRESSION_ITERATION_TIMES,
                                        predictionCol='DirPrediction')
                self._model[self.CHANGE_DIRECTION] = lr.fit(df)
            elif self._train_method[self.CHANGE_DIRECTION] == self.RANDOM_FOREST:
                rfc = RandomForestClassifier(featuresCol="features", labelCol=self.CHANGE_DIRECTION,
                                             numTrees=RANDOM_FOREST_TREE_NUMBER,
                                             maxDepth=RANDOM_FOREST_DEPTH, predictionCol='DirPrediction')
                self._model[self.CHANGE_DIRECTION] = rfc.fit(df)

            elif self._train_method[self.CHANGE_DIRECTION] == self.ARTIFICIAL_NEURAL_NETWORK:
                ann_layers[-1] = 2
                mlpc = MultilayerPerceptronClassifier(featuresCol="features", labelCol=self.CHANGE_DIRECTION,
                                                      layers=ann_layers,
                                                      predictionCol='DirPrediction')
                self._model[self.CHANGE_DIRECTION] = mlpc.fit(df)

            else:
                self.logger.warn('Unsupported training method {}'.format(self._train_method))
                raise ValueError('Unsupported training method {}'.format(self._train_method))

        else:
            if self._train_method == self.LINEAR_REGRESSION:
                lr = LinearRegression(featuresCol="features", labelCol=self.TARGET_PRICE, predictionCol='prediction',
                                      maxIter=LINEAR_REGRESSION_ITERATION_TIMES)
                self._model = lr.fit(df)
            elif self._train_method == self.RANDOM_FOREST:
                rfr = RandomForestRegressor(featuresCol="features", labelCol=self.TARGET_PRICE,
                                            predictionCol='prediction', numTrees=RANDOM_FOREST_TREE_NUMBER,
                                            maxDepth=RANDOM_FOREST_DEPTH)
                self._model = rfr.fit(df)

            elif self._train_method == self.ARTIFICIAL_NEURAL_NETWORK:
                ann_layers[-1] = 1
                self._model = KerasNeuralNetworkSpark(layers=ann_layers, spark=self._spark,
                                                      num_workers=WORKER_NUMBERS, epoch=100,
                                                      featuresCol="features", labelCol=self.TARGET_PRICE,
                                                      predictionCol='prediction'
                                                      )
                self._model.fit(df)

            else:
                self.logger.warn('Unsupported training method {}'.format(self._train_method))
                raise ValueError('Unsupported training method {}'.format(self._train_method))

        return self._model

    def _prepare_data_spark(self, data):
        """ Prepare data for spark format, output data will have the feature format and other useful information """

        keys = list(data.keys().difference({self.CHANGE_AMOUNT, self.CHANGE_DIRECTION, self.TARGET_PRICE,
                                            self.TODAY_PRICE}))

        df = self._spark.createDataFrame(data)
        ass = VectorAssembler(inputCols=keys, outputCol="features")
        output = ass.transform(df)
        # output.select('features', 'ChangeDirection', 'ChangeAmount').write.save('test.parquet')
        return output
