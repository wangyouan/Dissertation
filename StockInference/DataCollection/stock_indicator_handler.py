#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: stock_indicator_handler
# Author: Mark Wang
# Date: 1/6/2016

import numpy as np
from talib import abstract

from StockInference.DataCollection.base_class import BaseClass
from StockInference.util.get_history_stock_price import get_all_data_about_stock


class StockIndicatorHandler(BaseClass):
    def __init__(self):
        BaseClass.__init__(self)
        self._data_num = None

    def handle_indicator(self, required_info):
        self._data_num = len(self._stock_price)
        indicator_data = np.array([])
        get_data = None
        for info, info_parameter in required_info:
            if info == self.MACD:
                get_data = self.get_MACD(slow_period=info_parameter[self.MACD_SLOW_PERIOD],
                                         fast_period=info_parameter[self.MACD_FAST_PERIOD],
                                         signal_period=info_parameter[self.MACD_TIME_PERIOD])
            elif info == self.SMA:
                get_data = self.get_SMA(sma_period=info_parameter)
            elif info == self.EMA:
                get_data = self.get_EMA(ema_period=info_parameter)
            elif info == self.RSI:
                get_data = self.get_RSI(time_period=info_parameter)
            elif info == self.ROC:
                get_data = self.get_ROC(time_period=info_parameter)

            if get_data is not None:
                if not indicator_data:
                    indicator_data = np.atleast_2d(get_data).T
                else:
                    indicator_data = np.concatenate((indicator_data, np.atleast_2d(get_data).T), axis=1)

        return indicator_data.tolist()

    def get_ROC(self, time_period):
        data = self._get_stock_price_data(time_period)
        return abstract.ROC(data, timeperiod=time_period)[-self._data_num:]

    def get_RSI(self, time_period):
        data = self._get_stock_price_data(time_period)
        return abstract.RSI(data, timeperiod=time_period)[-self._data_num:]

    def get_EMA(self, ema_period, price=None):
        data = self._get_stock_price_data(ema_period - 1)
        return abstract.EMA(data, timeperiod=ema_period, price=price)[-self._data_num:]

    def get_SMA(self, sma_period, price=None):
        data = self._get_stock_price_data(sma_period - 1)
        if price is None:
            price = self.STOCK_CLOSE
        return abstract.SMA(data, timeperiod=sma_period, price=price)[-self._data_num:]

    def get_MACD(self, slow_period, fast_period, signal_period):
        additional_date = fast_period + signal_period
        data = self._get_stock_price_data(additional_date)
        macd, macd_signal, macd_hist = abstract.MACD(data, fast_period, slow_period, signal_period)
        return macd[-self._data_num:]

    def _get_stock_price_data(self, ahead_days):
        new_start_date = self.get_ahead_date(self._start_date, ahead_days)
        ahead_data = get_all_data_about_stock(self._stock_symbol, start_date=new_start_date,
                                              end_date=self.get_start_date())[:-1]
        ahead_data.extend(self._stock_price)
        remove_date = [i[1:] for i in ahead_data]
        return self._transform_data_to_talib_input(remove_date)

    def _transform_data_to_talib_input(self, data):
        np_array = np.array(data).astype(np.float)
        inputs = {
            self.STOCK_OPEN: np_array[:, 0],
            self.STOCK_HIGH: np_array[:, 1],
            self.STOCK_LOW: np_array[:, 2],
            self.STOCK_CLOSE: np_array[:, 3],
            self.STOCK_VOLUME: np_array[:, 4]
        }
        return inputs
