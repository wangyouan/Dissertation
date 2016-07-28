#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: __init__.py
# Author: Mark Wang
# Date: 24/7/2016

from stockforecaster.datacollect.base_class import BaseClass
from stockforecaster.util.date_util import *
from stockforecaster.util.query_yahoo_finance import *


class DataCollect(BaseClass):
    def __init__(self, stock_symbol, logger=None, data_path=None):
        BaseClass.__init__(self, logger=logger, data_dir_path=data_path, stock_symbol=stock_symbol)

    def get_required_data(self, required_info, start_date, end_date):
        self.logger.info("Start to collect data")
        self.set_end_date(end_date)
        self.set_start_date(start_date)
        self.set_is_adjusted(required_info[self.PRICE_TYPE] == self.STOCK_ADJUSTED_CLOSED)

        stock_price = self.load_historical_stock_price()

        if self.TECHNICAL_INDICATOR in required_info:
            indicator_df = self.get_indicator(required_info[self.TECHNICAL_INDICATOR])

    def load_historical_stock_price(self, start_date=None, end_date=None, adjusted=None):
        if start_date is None:
            start_date = self.get_start_date()

        if end_date is None:
            end_date = self.get_end_date()

        if adjusted is None:
            adjusted = self.get_is_adjusted()

        self.logger.info('Start to load historical stock price data')
        df = self._load_data_from_file(self.STOCK_PRICE, self._stock_symbol)
        if df is None or df.ix[0, self.DATE] > start_date or df.ix[df.shape[0] - 1, self.DATE] < end_date:
            self.logger.warn('No previous file founded or previous file data is not enough, '
                             'will load from yahoo finance')
            df = get_yahoo_finance_data(self._stock_symbol, remove_zero_volume=True)

        period_result = df[df[self.DATE] <= end_date][df[self.DATE] >= start_date]
        if adjusted:
            ratio = period_result[self.STOCK_ADJUSTED_CLOSED] / period_result[self.STOCK_CLOSE]
            period_result[self.STOCK_OPEN] *= ratio
            period_result[self.STOCK_HIGH] *= ratio
            period_result[self.STOCK_LOW] *= ratio
            period_result[self.STOCK_CLOSE] = period_result[self.STOCK_ADJUSTED_CLOSED]
        return period_result

    def get_indicator(self, indicator_list, start_date=None, end_date=None, adjusted=None):
        if start_date is None:
            start_date = self.get_start_date()

        if end_date is None:
            end_date = self.get_end_date()

        if adjusted is None:
            adjusted = self.get_is_adjusted()

        for info in indicator_list:
            pass


    def _handle_indicator_information(self, indicator_name, parameters):
        pass