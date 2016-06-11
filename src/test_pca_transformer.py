#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: test_pca_transformer
# Author: Mark Wang
# Date: 11/6/2016

import pickle

from StockInference.DataParser.data_parser import DataParser

f = open("/Users/warn/PycharmProjects/Dissertation/StockInference/test")
raw_data = pickle.load(f)
f.close()

dp = DataParser(None)
train, test, feature = dp.split_train_test_data(1, raw_data)
print train[0]
print len(train[0].features)

import test
test.test_file()