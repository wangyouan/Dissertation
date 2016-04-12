#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: linear_regression
# Author: Mark Wang
# Date: 12/4/2016

from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD

sc = SparkContext(appName="LinearRegressionPredict")


logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org").setLevel( logger.Level.OFF )
logger.LogManager.getLogger("akka").setLevel( logger.Level.OFF )


def prepare_data(path=r'../data/0001.HK.csv'):
    f = open(path)
    data_str = f.read()
    f.close()
    data_list = [map(float, i.split(',')[1:5]) for i in data_str.split('\n')[1:]]
    data_list = list(reversed(data_list))
    close_train_list = []
    data_len = len(data_list)
    train_data_len = int(data_len * 0.9)
    for i in xrange(1, train_data_len):
        close_price = data_list[i + 1][3]
        variable = data_list[i]
        close_train_list.append(LabeledPoint(features=variable, label=close_price))

    train_data = sc.parallelize(close_train_list)
    # print train_data.collect()
    model = LinearRegressionWithSGD.train(train_data, step=0.0001)

    test_data_list = []
    for i in xrange(train_data_len, data_len - 1):
        close_price = data_list[i + 1][3]
        variable = data_list[i]
        test_data_list.append(LabeledPoint(features=variable, label=close_price))

    test_data = sc.parallelize(test_data_list)
    value_predict = test_data.map(lambda p: (p.label, model.predict(p.features)))
    MSE = value_predict.map(lambda (v, p): (v - p) ** 2).reduce(lambda x, y: x + y) / value_predict.count()
    print("Mean Squared Error = " + str(MSE))
    print "Model coefficients:", str(model)
    print value_predict.collect()


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


if __name__ == "__main__":
    prepare_data()
    sc.stop()
