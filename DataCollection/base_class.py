#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: base_class
# Author: Mark Wang
# Date: 1/6/2016

import datetime

from constant import Constant


class BaseClass(Constant):
    def __init__(self):
        self._stock_price = []
        self._removed_date = []
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
