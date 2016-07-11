#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: data_parse
# Author: Mark Wang
# Date: 3/6/2016

import math


def min_max_de_normalize(label, features):
    max_price = features[1]
    min_price = features[2]
    return label * (max_price - min_price) / 2 + (max_price + min_price) / 2


def min_max_normalize(price, max_price, min_price):
    if abs(max_price - min_price) < 1e-4:
        return 0
    return (2 * price - (max_price + min_price)) / (max_price - min_price)


def get_MSE(label_and_prediction):
    """ Mean Squared Error """

    if hasattr(label_and_prediction, 'map'):
        return label_and_prediction.map(lambda (v, p): (v - p) * (v - p)).sum() / float(label_and_prediction.count())
    else:
        return sum(map(lambda (v, p): (v - p) * (v - p), label_and_prediction)) / float(len(label_and_prediction))


def get_MAD(label_prediction):
    """ Mean Absolute DEVIATION """

    if hasattr(label_prediction, 'map'):
        return label_prediction.map(lambda (v, p): abs(v - p)).sum() / float(label_prediction.count())
    else:
        return sum(map(lambda (v, p): abs(v - p), label_prediction)) / float(len(label_prediction))


def get_ME(label_prediction):
    """ Mean Absolute DEVIATION """

    if hasattr(label_prediction, 'map'):
        return label_prediction.map(lambda (v, p): v - p).sum() / float(label_prediction.count())
    else:
        return sum(map(lambda (v, p): v - p, label_prediction)) / float(len(label_prediction))


def get_MAPE(label_prediction):
    """ Mean Absolute Percentage Error """
    if hasattr(label_prediction, 'map'):
        return label_prediction.map(lambda (v, p): abs((v - p) / float(v))).sum() / float(label_prediction.count())
    else:
        return sum(map(lambda (v, p): abs((v - p) / float(v)), label_prediction)) / float(len(label_prediction))


def get_RMSE(label_and_prediction):
    """ Root Mean Squared Error """
    return math.sqrt(get_MSE(label_and_prediction))


def get_HMSE(label_and_prediction):
    if hasattr(label_and_prediction, 'map'):
        return label_and_prediction.map(lambda (v, p): (v / p - 1) ** 2).sum() / float(label_and_prediction.count())
    else:
        return sum(map(lambda (v, p): (v / p - 1), label_and_prediction)) / float(len(label_and_prediction))


def get_theils_inequality_coefficient(label_and_prediction):
    mse = get_MSE(label_and_prediction) * label_and_prediction.count()
    prediction = label_and_prediction.map(lambda (v, p): p ** p).sum() / float(label_and_prediction.count())
    label = label_and_prediction.map(lambda (v, p): v ** v).sum() / float(label_and_prediction.count())
    return mse / (math.sqrt(prediction) + math.sqrt(label))


# def get_CDC(label_prediction):
#     """ Correct Directional Change """
#     data = label_prediction.collect()
#     data_num = label_prediction.count()
#     correct_num = 0
#     for i in range(1, data_num):
#         label_change = data[i][0] - data[i - 1][0]
#         prediction = data[i][1] - data[i - 1][1]
#         if prediction * label_change > 0:
#             correct_num += 1
#
#     return correct_num * 100.0 / data_num


def get_CDC_combine(label_prediction):
    """ Correct Directional Change """
    if hasattr(label_prediction, 'collect'):
        data = label_prediction.collect()
        data_num = label_prediction.count()
    else:
        data = label_prediction
        data_num = len(label_prediction)
    correct_num = 0
    for i in range(1, data_num):
        label_change = data[i][0] - data[i - 1][0]
        prediction = data[i][1] - data[i - 1][0]
        if prediction * label_change > 0:
            correct_num += 1

    return float(correct_num) / float(data_num - 1)


get_CDC = get_CDC_combine
