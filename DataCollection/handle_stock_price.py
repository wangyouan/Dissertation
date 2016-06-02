#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: handle_stock_price
# Author: Mark Wang
# Date: 1/6/2016

import numpy

from base_class import BaseClass
from get_history_stock_price import get_all_data_about_stock


class StockPriceHandler(BaseClass):
    def __init__(self):
        BaseClass.__init__(self)

    def handle_stock_price(self, data_period):
        stock_info = [i[1:] for i in self._stock_price]
        if data_period > 1:
            append_start_date = self.get_ahead_date(self._start_date, (data_period - 1) * 5)
            append_end_date = self.get_ahead_date(self._start_date, 1)
            append_stock_price = get_all_data_about_stock(self._stock_symbol, append_start_date, append_end_date)[
                                 -(data_period - 1):]
            append_stock_info = [i[1:] for i in append_stock_price]
            append_stock_info.extend(stock_info)
            stock_info = append_stock_info

        data_num = len(stock_info)
        window_data = [stock_info[i:(i + data_period)] for i in range(data_num - data_period)]
        max_price = []
        min_price = []
        close_avg_price = []
        open_avg_price = []
        for window in window_data:
            max_price.append(max([i[1] for i in window]))
            min_price.append(min([i[2] for i in window]))
            close_avg_price.append(numpy.mean([i[3] for i in window]))
            open_avg_price.append(numpy.mean([i[0] for i in window]))
        features = zip(open_avg_price, max_price, min_price, close_avg_price)
        return features

    def normalized_label(self, normalized_method, label_list, price_list):
        new_label_list = []
        for i, label in enumerate(label_list):
            new_label_list.append(normalized_method(label, price_list[i][1], price_list[i][2]))
        return new_label_list
