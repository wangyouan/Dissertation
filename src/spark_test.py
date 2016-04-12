#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: spark_test
# Author: Mark Wang
# Date: 12/4/2016


from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD

sc = SparkContext(appName="LinearRegressionPredict")

logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org").setLevel(logger.Level.OFF)
logger.LogManager.getLogger("akka").setLevel(logger.Level.OFF)


def test_spark():
    def parsePoint(line):
        values = [float(x) for x in line.replace(',', ' ').split(' ')]
        return LabeledPoint(values[0], values[1:])

    data = sc.textFile(r"/usr/local/Cellar/apache-spark/1.6.1/libexec/data/mllib/ridge-data/lpsa.data")
    parsedData = data.map(parsePoint)
    print parsedData.collect()

    # Build the model
    model = LinearRegressionWithSGD.train(parsedData)

    # Evaluate the model on training data
    valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
    MSE = valuesAndPreds.map(lambda (v, p): (v - p) ** 2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
    print("Mean Squared Error = " + str(MSE))
    print "Model coefficients:", str(model)