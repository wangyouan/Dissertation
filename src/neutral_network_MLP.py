#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: neutral_network_MLP
# Author: Mark Wang
# Date: 17/4/2016

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from parse_data import DataParser
from constant import *

sc = SparkContext(appName="MLPNeutralNetwork")
sql_context = SQLContext(sc)

# Close logger
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org").setLevel(logger.Level.OFF)
logger.LogManager.getLogger("akka").setLevel(logger.Level.OFF)


def price_predict(path, windows=5):
    input_data = DataParser(path=path, window_size=windows)
