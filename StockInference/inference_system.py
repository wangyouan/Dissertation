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
            self.STOCK_PRICE: {self.DATA_PERIOD: 5},
            self.FUNDAMENTAL_ANALYSIS: [self.US10Y_BOND, self.US30Y_BOND, self.HSI, self.FXI, self.IC, self.IA]
        }
        calculated_data = self.data_collection.get_all_required_data(start_date=start_date, end_date=end_date,
                                                                     label_info=self.STOCK_CLOSE,
                                                                     normalized_method=self.MIN_MAX,
                                                                     required_info=required_info)
        import pprint
        pprint.pprint(calculated_data, width=120)


if __name__ == "__main__":
    test = InferenceSystem('0001.HK')
    test.predict_historical_data(0.8, "2015-06-01", "2015-07-01")