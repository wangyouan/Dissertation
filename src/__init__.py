#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: __init__.py
# Author: Mark Wang
# Date: 8/4/2016

import logging

try:
    from pyspark import SparkContext
    from pyspark.sql import SQLContext
    from pyspark import SparkConf
except ImportError, e:
    logger = logging.getLogger(__name__)
    logger.warning("Cannot load spark, as {}".format(e))
    SparkContext = None
    SQLContext = None
    SparkConf = None


def load_spark_context(application_name=None):
    if application_name is None:
        application_name = __name__

    conf = SparkConf().setAppName(application_name)
    sc = SparkContext.getOrCreate(conf=conf)
    sql_context = SQLContext(sc)

    # Close logger
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
    return sc, sql_context


def load_logger(application_name="", level=None):
    logger = logging.getLogger(application_name)
    if level is not None:
        logger.setLevel(level=level)
    return logger