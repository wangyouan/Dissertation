#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: __init__.py
# Author: Mark Wang
# Date: 24/7/2016

import pandas as pd
from talib import abstract

from stockforecaster.datacollect.base_class import BaseClass
from stockforecaster.util.query_yahoo_finance import get_yahoo_finance_data


class DataCollect(BaseClass):
    def __init__(self, stock_symbol, logger=None, data_path=None):
        BaseClass.__init__(self, logger=logger, data_dir_path=data_path, stock_symbol=stock_symbol)

    def get_required_data(self, required_info, start_date, end_date):
        self.logger.info("Start to collect data")
        self.set_end_date(end_date)
        self.set_start_date(start_date)
        self.set_is_adjusted(required_info[self.PRICE_TYPE] == self.STOCK_ADJUSTED_CLOSED)

        stock_price = self.load_historical_stock_price(self.get_start_date(), self.get_end_date(),
                                                       self.get_is_adjusted())

        if self.TECHNICAL_INDICATOR in required_info:
            indicator_df = self.get_indicator(required_info[self.TECHNICAL_INDICATOR])

    def load_historical_stock_price(self, start_date=None, end_date=None, adjusted=None):

        if adjusted is None:
            adjusted = self.get_is_adjusted()

        # when no date is specified, then will return all data that could be downloaded from Yahoo finance
        if start_date is None and end_date is None:
            period_result = get_yahoo_finance_data(self._stock_symbol, remove_zero_volume=True)
        else:
            self.logger.info('Start to load historical stock price data')
            data_df = self._load_data_from_file(self.STOCK_PRICE, self._stock_symbol)
            if data_df is None or (start_date is not None and data_df.index.min() > start_date) \
                    or (end_date is not None and data_df.index.max() < end_date):
                self.logger.warn('No previous file founded or previous file data is not enough, '
                                 'will load from yahoo finance')
                data_df = get_yahoo_finance_data(self._stock_symbol, remove_zero_volume=True)
                self._save_data_to_file(data_df, self._stock_symbol, self.STOCK_PRICE)

            if start_date is None:
                period_result = data_df[data_df.index < end_date]
            elif end_date is None:
                period_result = data_df[data_df.index > start_date]

            else:
                period_result = data_df[data_df.index > start_date]
                period_result = period_result[period_result.index < end_date]

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

        info_df = pd.DataFrame()
        for info in indicator_list:
            info_df = pd.concat([info_df, self._handle_indicator_information(info[0], info[1])], axis=1)

        return info_df

    def _handle_indicator_information(self, indicator_name, parameters):
        stock_price_df = self.load_historical_stock_price()
        if self.get_is_adjusted():
            stock_price_df['close'] = stock_price_df[self.STOCK_ADJUSTED_CLOSED]
            ratio = stock_price_df[self.STOCK_ADJUSTED_CLOSED] / stock_price_df[self.STOCK_CLOSE]
            stock_price_df['open'] = stock_price_df[self.STOCK_OPEN] * ratio
            stock_price_df['high'] = stock_price_df[self.STOCK_HIGH] * ratio
            stock_price_df['low'] = stock_price_df[self.STOCK_LOW] * ratio
            stock_price_df['volume'] = stock_price_df[self.STOCK_VOLUME] * ratio
        else:
            stock_price_df['close'] = stock_price_df[self.STOCK_CLOSE]
            stock_price_df['open'] = stock_price_df[self.STOCK_OPEN]
            stock_price_df['high'] = stock_price_df[self.STOCK_HIGH]
            stock_price_df['low'] = stock_price_df[self.STOCK_LOW]
            stock_price_df['volume'] = stock_price_df[self.STOCK_VOLUME]
