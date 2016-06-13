#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: data_parse
# Author: Mark Wang
# Date: 3/6/2016

import pickle


def min_max_de_normalize(label, features):
    max_price = features[1]
    min_price = features[2]
    return label * (max_price - min_price) / 2 + (max_price + min_price) / 2


def min_max_normalize(price, max_price, min_price):
    if abs(max_price - min_price) < 1e-4:
        return 0
    return (2 * price - (max_price + min_price)) / (max_price - min_price)


def get_MSE(label_and_prediction):
    return label_and_prediction.map(lambda (v, p): (v - p) * (v - p)).sum() / float(label_and_prediction.count())


def get_MAD(label_prediction):
    return label_prediction.map(lambda (v, p): abs(v - p)).sum() / float(label_prediction.count())


def get_MAPE(label_prediction):
    return label_prediction.map(lambda (v, p): abs((v - p) / float(v))).sum() / float(label_prediction.count())
