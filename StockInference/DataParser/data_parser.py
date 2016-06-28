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


def ratio_function(data):
    return abs(float(data[0] - data[1])) / data[1]


def raw_difference(data):
    return abs(data[0] - data[1])


def direction(data):
    return int(data[0] > data[1])


class DataParser(Constants):
    def __init__(self, n_components=None, label_data_type=None):
        self.first_transformer = get_transformer(1)
        self.second_transformer = get_transformer(n_components)
        self.third_transformer = get_transformer(1)
        self.label_data_type = label_data_type
        self.label_transformer = None

    def split_train_test_data(self, train_num, raw_data, fit_transform=False):
        """
            Base on the train ratio, will split raw data into 3 parts, one is train (normalized_label with transformed
            feature), another is test (non-normalized_label with transformed feature) the third one is the
            non-transformed test features
        """
        total_num = len(raw_data)
        if self.label_data_type is None or self.label_data_type == self.ORIGINAL_PRICE:
            train_raw_data = raw_data[:train_num]
            test_raw_data = raw_data[train_num:]
            test_raw_features = [i.features[:4] for i in test_raw_data]

            normalized_label = map(
                lambda p: LabeledPoint(features=p.features,
                                       label=min_max_normalize(p.label, p.features[1], p.features[2])),
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
        elif self.label_data_type == self.RAW_AMOUNT or self.label_data_type == self.RATIO_AMOUNT:
            tomorrow_today = map(lambda p: (p.label, p.features[3]), raw_data)
            amount_label = map(ratio_function if self.label_data_type == self.RATIO_AMOUNT else raw_difference,
                               tomorrow_today)
            trend_label = map(direction, tomorrow_today)
            raw_features = [p.features for p in raw_data]
            train_features = raw_features[:train_num]
            test_features = raw_features[train_num:]
            train_trend_label = trend_label[:train_num]
            test_trend_label = trend_label[train_num:]
            train_amount_label = amount_label[:train_num]
            test_amount_label = amount_label[train_num:]
            if fit_transform:
                train_features = self.fit_transform(train_features)
            else:
                train_features = self.transform(train_features)
            test_features = self.transform(test_features)
            train_amount_label = np.array(train_amount_label).reshape((1, -1)).T
            test_amount_label = np.array(test_amount_label).reshape((1, -1)).T
            train_amount_label = self.transform_label(train_amount_label).reshape((train_num,))
            test_amount_label = self.transform_label(test_amount_label).reshape((total_num - train_num,))
            train_trend = [LabeledPoint(label=i, features=j) for i, j in zip(train_trend_label, train_features)]
            train_amount = [LabeledPoint(label=i, features=j) for i, j in zip(train_amount_label, train_features)]
            test_trend = [LabeledPoint(label=i, features=j) for i, j in zip(test_trend_label, test_features)]
            test_amount = [LabeledPoint(label=i, features=j) for i, j in zip(test_amount_label, test_features)]
            return [train_trend, train_amount], [test_trend, test_amount], tomorrow_today

    def inverse_transform_label(self, label_list):
        if hasattr(label_list, '__len__'):
            a = np.array(label_list)
            num = len(label_list)
            a = a.reshape((1, -1)).T
            return self.label_transformer.inverse_transform(a).reshape((num,))
        else:
            a = np.array([label_list]).reshape((1,-1))
            return self.label_transformer.inverse_transform(a)[0][0]

    def transform_label(self, label_list):
        if self.label_transformer is None:
            self.label_transformer = MinMaxScaler(feature_range=(-1, 1))
            return self.label_transformer.fit_transform(label_list)
        else:
            return self.label_transformer.transform(label_list)

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
        if isinstance(raw_data[0], LabeledPoint):
            raw_features = [np.array(p.features) for p in raw_data]
            first = self.first_transformer.fit_transform(raw_features)
            second = self.second_transformer.fit_transform(first)
            final = self.third_transformer.fit_transform(second)

            return [LabeledPoint(label=i.label, features=j) for i, j in zip(raw_data, final)]
        else:
            first = self.first_transformer.fit_transform(raw_data)
            second = self.second_transformer.fit_transform(first)
            final = self.third_transformer.fit_transform(second)
            return final


if __name__ == "__main__":
    dp = DataParser()
    f = open('test', 'w')
    import pickle

    pickle.dump(dp, f)
    f.close()
