#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: parse_data
# Author: Mark Wang
# Date: 17/4/2016


class DataParser(object):
    """
    Use to handle data

    self.__path: CRV
    """
    def __init__(self, path, window_size=1):
        self.__path = path
        self.__days = window_size

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
        time_series_data = self.__get_time_series_data(data_list=data_list, window_size=n_days)

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
        close_data
