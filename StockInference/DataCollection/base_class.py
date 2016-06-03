#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: base_class
# Author: Mark Wang
# Date: 1/6/2016

import datetime
import logging

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

    def _merge_info(self, calculated_info, info_dict):

        # merge bond info into calculated list
        for i in calculated_info:
            if i[0] in info_dict:
                i.append(info_dict[i[0]])
            else:
                i.append(0)
                self._removed_date.add(i[0])
        return calculated_info
