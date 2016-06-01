#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: base_class
# Author: Mark Wang
# Date: 1/6/2016

from constant import Constant


class BaseClass(Constant):
    def __init__(self):
        self._stock_price = []
        self._start_date = None
        self._end_date = None
        self._true_end_date = None
        self._date_list = None
