#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: parameters
# Author: Mark Wang
# Date: 19/10/2016


class Parameters(object):
    # for linear regression
    linear_regression_training_times = 100000
    linear_regression_regularization_parameter = 0.01

    # for logistic_regression
    logistic_regression_training_times = 100000
    logistic_regression_regularization_parameter = 0.01

    # for random forest
    random_forest_tree_number = 30
    random_forest_tree_max_depth = 20

    # neural network
    ann_epoch_number = 100

    # spark parameters:
    spark_worker_numbers = 2
    # spark_driver_memory = '1g'
    # spark_executor_memory = '1g'
