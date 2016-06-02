#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: stock_indicator_handler
# Author: Mark Wang
# Date: 1/6/2016

from base_class import BaseClass

class StockIndicatorHandler(BaseClass):
    def __init__(self):
        BaseClass.__init__(self)
        self._append_data = []

    def handle_indicator(self, required_info):
        pass
