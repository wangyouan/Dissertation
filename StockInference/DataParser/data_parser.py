#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: data_parser
# Author: Mark Wang
# Date: 5/6/2016

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyspark.mllib.regression import LabeledPoint

from StockInference.constant import Constants
from StockInference.util.data_parse import min_max_normalize


class Fun():
    def fit_transform(self, raw_data):
        return raw_data

    def transform(self, raw_data):
        return raw_data


def get_transformer(transformer_type):
    if transformer_type == 1:
        return MinMaxScaler(feature_range=(-1, 1))
    elif transformer_type == 2:
        return StandardScaler()
    elif transformer_type == 3:
        return Fun()
    else:
        return PCA(transformer_type)


class DataParser(Constants):
    def __init__(self, n_components=None):
        self.first_transformer = get_transformer(1)
        self.second_transformer = get_transformer(n_components)
        self.third_transformer = get_transformer(1)

    def split_train_test_data(self, train_ratio, raw_data, fit_transform=False):
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
            lambda p: LabeledPoint(features=p.features, label=min_max_normalize(p.label, p.features[1], p.features[2])),
            train_raw_data
        )
        if fit_transform:
            train_transformed = self.fit_transform(normalized_label)
        else:
            train_transformed = self.transform(normalized_label)

        if test_raw_data:
            test_transformed = self.transform(test_raw_data)
        else:
            test_transformed = []
        return train_transformed, test_transformed, test_raw_features

    def transform(self, raw_data):
        if isinstance(raw_data[0], LabeledPoint):
            raw_features = [np.array(p.features) for p in raw_data]
            first = self.first_transformer.transform(raw_features)
            second = self.second_transformer.transform(first)
            final = self.third_transformer.transform(second)
            return [LabeledPoint(label=i.label, features=j) for i, j in zip(raw_data, final)]
        else:
            first = self.first_transformer.transform(raw_data)
            second = self.second_transformer.transform(first)
            final = self.third_transformer.transform(second)
            return final

    def fit_transform(self, raw_data):
        raw_features = [np.array(p.features) for p in raw_data]
        first = self.first_transformer.fit_transform(raw_features)
        second = self.second_transformer.fit_transform(first)
        final = self.third_transformer.fit_transform(second)

        return [LabeledPoint(label=i.label, features=j) for i, j in zip(raw_data, final)]


if __name__ == "__main__":
    dp = DataParser()
    f = open('test', 'w')
    import pickle
    pickle.dump(dp, f)
    f.close()
