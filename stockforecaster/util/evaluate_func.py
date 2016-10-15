#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: evaluate_func
# Author: Mark Wang
# Date: 15/10/2016

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_mean_squared_error(df, result_col, target_col):
    mse = mean_squared_error(df[target_col], df[result_col])
    return mse


def calculate_rooted_mean_squared_error(df, result_col, target_col):
    rmse = np.sqrt(calculate_mean_squared_error(df, result_col, target_col))
    return rmse


def calculate_mean_absolute_error(df, result_col, target_col):
    mae = mean_absolute_error(df[target_col], df[result_col])
    return mae


def calculate_success_direction_prediction_rate(df, today, result_col, target_col):
    actual = (df[target_col] - df[today]) > 0
    prediction = (df[result_col] - df[today]) > 0
    return float(sum(actual == prediction)) / float(df.shape[0])


def calculate_mean_absolute_percentage_error(df, result_col, target_col):
    actual = df[result_col]
    prediction = df[target_col]
    error = actual - prediction
    mape = (error.apply(abs) / actual).sum() / float(df.shape[0])
    return mape
