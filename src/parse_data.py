#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: parse_data
# Author: Mark Wang
# Date: 17/4/2016


def avg(data_list):
    """
    Use to calculate simple moving average
    :param data_list: the list need to be calculated
    :return: the moving average of the data list
    """
    return sum(data_list) / float(len(data_list))


class DataParser(object):
    """
    Use to handle data

    self.__path: CRV
    """
    def __init__(self, path, window_size=1):
        self.__path = path
        self.__days = window_size
        self.features = None

    def get_n_days_history_data(self, data_type="DataFrame", n_days=None):
        """
        Use to handle yahoo finance history data, return will be DataFrame of RDD
        :param window_size: get how many days
        :param data_type: "DataFrame" or "RDD"
        :param required_list: which part of data is needed [Open,High,Low,Close,Volume,Adj Close]
        :return: Required two required data, label normalized close price, features: [highest price, lowest price,
                 average close price]
        """
        data_list = self.load_data_from_yahoo_csv()
        close_data, open_data, self.features = self.__get_time_series_data(data_list=data_list, window_size=n_days)

    def normalize_data(self, close_data=None, open_data=None, features=None):
        if features is None:
            features = self.features

        data_len = len(features)

        def normalize(price, max_price, min_price):
            return ((2 * price - (max_price + min_price)) / (max_price - min_price))

        close_normalize_data = []
        open_normalize_data = []
        for i in range(data_len):
            if close_data:
                close_price = normalize(close_data[i], features[i][1], features[i][2])
                close_normalize_data.append((features[i], close_price))

            if open_data:
                open_price = normalize(open_data[i], features[i][1], features[i][2])
                open_normalize_data.append((features[i], open_price))

        return open_normalize_data, close_normalize_data

    def de_normalize_data(self, close_data, open_data, features):
        def de_normalize(price, max_price, min_price):
            return (price * (max_price - min_price)) / 2 + (max_price + min_price) / 2

    def load_data_from_yahoo_csv(self, path=None):
        if path is None:
            path = self.__path

        f = open(path)
        data_str = f.read()
        f.close()
        data_list = [map(float, i.split(',')[1:]) for i in data_str.split('\n')[1:]]
        data_list.reverse()
        return data_list

    def __get_time_series_data(self, data_list, window_size=None):
        """
        Get time series from given data
        :param data_list: [open, high, low, close]
        :param window_size: the number of days
        :return:
        """
        if window_size is None:
            window_size = self.__days

        data_num = len(data_list)
        close_data = [data_list[i][3] for i in range(window_size, data_num)]
        open_data = [data_list[i][0] for i in range(window_size, data_num)]
        window_data = [data_list[i:(i + window_size)] for i in range(data_num - window_size)]

        # create feature prices
        max_price = []
        min_price = []
        close_avg_price = []
        open_avg_price = []
        for window in window_data:
            max_price.append(max([i[1] for i in window]))
            min_price.append(min([i[2] for i in window]))
            close_avg_price.append(avg([i[3] for i in window]))
            open_avg_price.append(avg([i[0] for i in window]))
        features = zip(open_avg_price, max_price, min_price, close_avg_price)

        return close_data, open_data, features
