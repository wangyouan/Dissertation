#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: __init__.py
# Author: Mark Wang
# Date: 24/7/2016

from stockforecater.datacollect.base_class import BaseClass


class DataCollect(BaseClass):
    def __init__(self, stock_symbol, logger=None, data_path=None):
        BaseClass.__init__(self, logger=logger, data_dir_path=data_path, stock_symbol=stock_symbol)

    def get_required_data(self, required_info):
        self.logger.info("Start to collect data")
