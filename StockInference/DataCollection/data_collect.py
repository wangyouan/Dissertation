#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: data_collect
# Author: Mark Wang
# Date: 30/5/2016

import math

from pyspark.mllib.regression import LabeledPoint

from StockInference.DataCollection.fundamental_analysis import FundamentalAnalysis
from StockInference.DataCollection.handle_stock_price import StockPriceHandler
from StockInference.DataCollection.stock_indicator_handler import StockIndicatorHandler
from StockInference.util.get_history_stock_price import get_all_data_about_stock


def normalize(price, max_price, min_price):
    if abs(max_price - min_price) < 1e-4:
        return 0
    return (2 * price - (max_price + min_price)) / (max_price - min_price)


class DataCollect(StockPriceHandler, StockIndicatorHandler, FundamentalAnalysis):
    def __init__(self, stock_symbol):
        StockIndicatorHandler.__init__(self)
        StockPriceHandler.__init__(self)
        FundamentalAnalysis.__init__(self)
        self._stock_symbol = stock_symbol

    def get_all_required_data(self, start_date, end_date, label_info, normalized_method, required_info):
        """
        Used to get all dict

        :param start_date: start date of the required info data type 2016-02-01
        :param end_date: end data of the required info data type 2016-02-01
        :param required_info: a dict contains all the required info
        :param label_info: using which price as label, only support CLOSE or OPEN
        :param normalized_method: using which method to do normalization a function takes in 3 parameters and return one
        :return: an RDD data structure based on the required info
        """
        if not self._stock_price or self._start_date != start_date or self._end_date != end_date:
            self._removed_date = set()
            self._start_date = start_date
            self._end_date = end_date
            self._stock_price = get_all_data_about_stock(symbol=self._stock_symbol, start_date=start_date,
                                                         end_date=end_date)
            self._date_list = [i[0] for i in self._stock_price[:-1]]
            self._true_end_date = self._date_list[-1]

        if normalized_method is None or normalized_method == self.MIN_MAX:
            normalized_method = normalize
        elif normalized_method == self.SIGMOID:
            normalized_method = lambda (x, y, z): 1.0 / (1 + math.exp(x))

        if label_info == self.STOCK_CLOSE:
            label_list = [i[4] for i in self._stock_price[1:]]
        else:
            label_list = [i[1] for i in self._stock_price[1:]]

        collected_data = self.handle_stock_price(required_info[self.STOCK_PRICE][self.DATA_PERIOD])
        label_list = self.normalized_label(normalized_method=normalized_method, label_list=label_list,
                                           price_list=collected_data)

        if self.STOCK_INDICATOR in required_info:
            indicator_info = self.handle_indicator(required_info[self.STOCK_INDICATOR])
            collected_data = [i + j for i, j in zip(collected_data, indicator_info)]

        if self.FUNDAMENTAL_ANALYSIS in required_info:
            fundamental_info = self.fundamental_analysis(required_info[self.FUNDAMENTAL_ANALYSIS])
            collected_data = [i + j for i, j in zip(collected_data, fundamental_info)]

        label_pointed_list = []
        for i, j in zip(collected_data, label_list):
            label_pointed_list.append(LabeledPoint(features=i, label=j))

        return label_pointed_list
