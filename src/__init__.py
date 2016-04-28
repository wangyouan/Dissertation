#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: __init__.py
# Author: Mark Wang
# Date: 8/4/2016

import logging
import sys

from pyspark import SparkContext
from pyspark.sql import SQLContext

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout)


def load_spark_context():
    sc = SparkContext(appName="MLPNeutralNetwork")
    sql_context = SQLContext(sc)

    # Close logger
    # logger = sc._jvm.org.apache.log4j
    # logger.LogManager.getLogger("org").setLevel(logger.Level.OFF)
    # logger.LogManager.getLogger("akka").setLevel(logger.Level.OFF)
    return sc, sql_context


def load_logger(application_name=""):
    logger = logging.getLogger(application_name)
    return logger
