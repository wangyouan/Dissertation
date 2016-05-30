#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: data_collect
# Author: Mark Wang
# Date: 30/5/2016

from constant import Constant

class DataCollect(Constant):
    def __init__(self):
        pass

    def get_all_required_data(self, stock_symbol, start_date, end_date, required_info):
        """
        Used to get all dict

        :param stock_symbol: symbol of the target stock
        :param start_date: start date of the required info
        :param end_date: end data of the required info
        :param required_info: a dict contains all the required info
        :return: an RDD data structure based on the required info
        """
        pass