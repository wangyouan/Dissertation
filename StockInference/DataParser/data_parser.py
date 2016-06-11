#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: data_parser
# Author: Mark Wang
# Date: 5/6/2016

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pyspark.mllib.regression import LabeledPoint

from StockInference.constant import Constants
from StockInference.util.data_parse import min_max_normalize


class DataParser(Constants):
    def __init__(self):
        self.data_transformer = None
        self.standard_scale = None

    def split_train_test_data(self, train_ratio, raw_data, n_components='mle'):
        """
            Base on the train ratio, will split raw data into 3 parts, one is train (normalized_label with transformed
            feature), another is test (non-normalized_label with transformed feature) the third one is the
            non-transformed test features
        """
        total_num = len(raw_data)
        train_num = int(train_ratio * total_num)
        train_raw_data = raw_data[:train_num]
        test_raw_data = raw_data[train_num:]
        test_raw_features = [i.features[:4] for i in test_raw_data]

        normalized_label = map(
            lambda p: LabeledPoint(features=p.features, label=min_max_normalize(p.label, p.features[1], p.features[0])),
            train_raw_data
        )
        train_transformed = self.fit_transform(normalized_label)

        test_transformed = self.transform(test_raw_data)
        return train_transformed, test_transformed, test_raw_features

    def transform(self, raw_data):
        raw_features = [np.array(p.features) for p in raw_data]
        after_pca = self.data_transformer.transform(raw_features)
        after_normalize = self.standard_scale.transform(after_pca)
        return [LabeledPoint(label=i.label, features=j) for i, j in zip(raw_data, after_normalize)]

    def fit_transform(self, raw_data, n_components=None):
        self.data_transformer = PCA(n_components=n_components)
        self.standard_scale = StandardScaler()
        raw_features = [np.array(p.features) for p in raw_data]
        after_pca = self.data_transformer.fit_transform(raw_features)
        after_stand = self.standard_scale.fit_transform(after_pca)

        return [LabeledPoint(label=i.label, features=j) for i, j in zip(raw_data, after_stand)]
