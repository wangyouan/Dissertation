#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: __init__.py
# Author: Mark Wang
# Date: 8/4/2016

from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext(appName="MLPNeutralNetwork")
sql_context = SQLContext(sc)

# Close logger
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org").setLevel(logger.Level.OFF)
logger.LogManager.getLogger("akka").setLevel(logger.Level.OFF)