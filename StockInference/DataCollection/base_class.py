#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: base_class
# Author: Mark Wang
# Date: 1/6/2016

import datetime

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

    @staticmethod
    def get_ahead_date(detailed_date, ahead_period):
        date_object = datetime.date.strptime(detailed_date, "%Y-%m-%d")
        date_object -= datetime.timedelta(ahead_period)
        return date_object.strftime("%Y-%m-%d")

    def _merge_info(self, calculated_info, info_dict):

        # merge bond info into calculated list
        for i in calculated_info:
            if i[0] in info_dict:
                i.append(info_dict[i[0]])
            else:
                i.append(0)
                self._removed_date.add(i[0])
        return calculated_info
