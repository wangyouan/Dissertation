#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: base_class
# Author: Mark Wang
# Date: 1/6/2016

import datetime
import logging

import pandas as pd
from sklearn.decomposition import PCA
from pandas.tseries.offsets import BDay

from StockInference.constant import Constants


class BaseClass(Constants):
    def __init__(self):
        self._stock_price = []
        self._removed_date = set()
        self._start_date = None
        self._end_date = None
        self._true_end_date = None
        self._date_list = None
        self._stock_symbol = None
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def get_ahead_date(detailed_date, ahead_period):
        date_list = detailed_date.split('-')
        date_object = datetime.date(year=int(date_list[0]), month=int(date_list[1]), day=int(date_list[2]))
        date_object -= datetime.timedelta(ahead_period)
        return date_object.strftime("%Y-%m-%d")

    def set_start_date(self, date):
        self._start_date = date

    def set_end_date(self, date):
        self._end_date = date
        self._true_end_date = date

    def generate_date_list(self):
        start_date = self._start_date.split('-')
        end_date = self._end_date.split('-')
        start_date = map(int, start_date)
        end_date = map(int, end_date)
        start_date = pd.datetime(year=start_date[0], month=start_date[1], day=start_date[2])
        end_date = pd.datetime(year=end_date[0], month=end_date[1], day=end_date[2])
        self._date_list = []
        while start_date <= end_date:
            self._date_list.append(start_date.strftime("%Y-%m-%d"))
            start_date += BDay(1)

    def _merge_info(self, calculated_info, info_dict):

        # merge bond info into calculated list
        for i in calculated_info:
            if i[0] in info_dict:
                i.append(info_dict[i[0]])
            else:
                i.append(0)
                self._removed_date.add(i[0])
        return calculated_info

    def get_pca_transformer(self, data, n_components=None):
        pca_transformer = PCA(n_components=n_components)
        pca_transformer.fit(data)
        return pca_transformer


if __name__ == "__main__":
    from data_collect import DataCollect

    dc = DataCollect("0001.HK")
    dc.set_start_date("2012-03-01")
    dc.set_end_date("2012-04-01")
    data = dc.fundamental_analysis([dc.US10Y_BOND, dc.US30Y_BOND, dc.HSI, dc.FXI, dc.IC, dc.IA])
