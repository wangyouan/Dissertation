#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: data_collect
# Author: Mark Wang
# Date: 30/5/2016


from pyspark.mllib.regression import LabeledPoint

from StockInference.DataCollection.fundamental_analysis import FundamentalAnalysis
from StockInference.DataCollection.handle_stock_price import StockPriceHandler
from StockInference.DataCollection.stock_indicator_handler import StockIndicatorHandler
from StockInference.util.get_history_stock_price import get_all_data_about_stock


class DataCollect(StockPriceHandler, StockIndicatorHandler, FundamentalAnalysis):
    def __init__(self, stock_symbol, start_date, end_date, data_file_path=None, logger=None):
        StockIndicatorHandler.__init__(self, logger)
        StockPriceHandler.__init__(self, logger)
        FundamentalAnalysis.__init__(self, logger)
        self._stock_symbol = stock_symbol
        self.set_start_date(start_date)
        self.set_end_date(end_date)
        self.set_data_file_path(data_file_path)

    def get_raw_data(self, label_info, required_info):

        self.logger.info("Start to collect data")
        self.set_price_type(label_info)

        stock_price = self.load_data_from_file("stock_price_history")
        if stock_price is None:
            self._stock_price = get_all_data_about_stock(symbol=self._stock_symbol, start_date=self.get_start_date(),
                                                         end_date=self.get_end_date(), remove_zero_volume=True)
            self.set_date_list([i[0] for i in self._stock_price[:-1]])
            self.save_data_to_file("stock_price_history", self._stock_price)
        else:
            self._stock_price = stock_price
            self.set_date_list([i[0] for i in self._stock_price[:-1]])

        if self.get_price_type() == self.STOCK_CLOSE:
            label_list = [i[4] for i in self._stock_price[1:]]
        elif self.get_price_type() == self.STOCK_OPEN:
            label_list = [i[1] for i in self._stock_price[1:]]
        else:
            label_list = [i[6] for i in self._stock_price[1:]]

        self.logger.info("Finish handle stock price")

        if self.STOCK_PRICE in required_info:
            self.logger.info("Start to handle features")
            collected_data = self.handle_stock_price(required_info[self.STOCK_PRICE][self.DATA_PERIOD])
        else:
            collected_data = [[i] for i in self.get_date_list()]

        if self.STOCK_INDICATOR in required_info:
            self.logger.info("Start to get technique indicators")
            indicator_info = self.handle_indicator(required_info[self.STOCK_INDICATOR])
            if len(indicator_info) < len(collected_data):
                collected_data = collected_data[-len(indicator_info):]
            collected_data = [i + j for i, j in zip(collected_data, indicator_info)]

        if self.FUNDAMENTAL_ANALYSIS in required_info:
            self.logger.info("Start to get some fundamental information")
            fundamental_info = self.fundamental_analysis(required_info[self.FUNDAMENTAL_ANALYSIS])
            if self._data_num < len(collected_data):
                fundamental_info = fundamental_info[-self._data_num:]
                collected_data = collected_data[-self._data_num:]
            collected_data = [i + j for i, j in zip(collected_data, fundamental_info)]

        label_pointed_list = []
        label_list = label_list[-len(collected_data):]
        for i, j in zip(collected_data, label_list):
            label_pointed_list.append(LabeledPoint(features=i, label=j))

        return label_pointed_list


if __name__ == "__main__":
    test = DataCollect('0001.HK', "2012-02-01", "2014-02-01")
