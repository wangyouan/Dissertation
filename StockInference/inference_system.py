#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: inference_system
# Author: Mark Wang
# Date: 1/6/2016

from constant import Constants
from data_collect import DataCollect


class InferenceSystem(Constants):
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        self.data_collection = DataCollect(stock_symbol=stock_symbol)

    def predict_historical_data(self, train_test_ratio, start_date, end_date):
        required_info = {
        }