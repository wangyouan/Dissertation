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
from StockInference.util.date_parser import get_ahead_date


class StockIndicatorHandler(BaseClass):
    def __init__(self, logger=None):
        BaseClass.__init__(self, logger=None)
        self._data_num = None

    def handle_indicator(self, required_info):
        self._data_num = len(self._stock_price)
        indicator_data = np.array([])
        get_data = None
        for info, info_parameter in required_info:
            if info == self.MACD:
                file_name = "MACD_{}_{}_{}.dat".format(info_parameter[self.MACD_SLOW_PERIOD],
                                                       info_parameter[self.MACD_FAST_PERIOD],
                                                       info_parameter[self.MACD_TIME_PERIOD])
                get_data = self.load_data_from_file(file_name)
                if get_data is None:
                    get_data = self.get_MACD(slow_period=info_parameter[self.MACD_SLOW_PERIOD],
                                             fast_period=info_parameter[self.MACD_FAST_PERIOD],
                                             signal_period=info_parameter[self.MACD_TIME_PERIOD])
                    self.save_data_to_file(file_name, get_data)
            elif info == self.SMA:
                file_name = "{}_{}.dat".format(info, info_parameter)
                get_data = self.load_data_from_file(file_name)
                if get_data is None:
                    get_data = self.get_SMA(period=info_parameter)
                    self.save_data_to_file(file_name, get_data)
            elif info == self.EMA:
                file_name = "{}_{}.dat".format(info, info_parameter)
                get_data = self.load_data_from_file(file_name)
                if get_data is None:
                    get_data = self.get_EMA(period=info_parameter)
                    self.save_data_to_file(file_name, get_data)
            elif info == self.RSI:
                file_name = "{}_{}.dat".format(info, info_parameter)
                get_data = self.load_data_from_file(file_name)
                if get_data is None:
                    get_data = self.get_RSI(period=info_parameter)
                    self.save_data_to_file(file_name, get_data)
            elif info == self.ROC:
                file_name = "{}_{}.dat".format(info, info_parameter)
                get_data = self.load_data_from_file(file_name)
                if get_data is None:
                    get_data = self.get_ROC(period=info_parameter)
                    self.save_data_to_file(file_name, get_data)

            if get_data is not None:
                if not indicator_data.any():
                    indicator_data = np.atleast_2d(get_data).T
                else:
                    indicator_data = np.concatenate((indicator_data, np.atleast_2d(get_data).T), axis=1)

        return indicator_data.tolist()

    def get_ROC(self, period):
        data = self._get_stock_price_data(period)
        return abstract.ROC(data, timeperiod=period)[-self._data_num:]

    def get_RSI(self, period):
        data = self._get_stock_price_data(period)
        return abstract.RSI(data, timeperiod=period)[-self._data_num:]

    def get_EMA(self, period, price=None):
        data = self._get_stock_price_data(period)
        if price is None:
            price = self.STOCK_CLOSE
        return abstract.EMA(data, timeperiod=period, price=price)[-self._data_num:]

    def get_SMA(self, period, price=None):
        data = self._get_stock_price_data(period)
        if price is None:
            price = self.STOCK_CLOSE
        return abstract.SMA(data, timeperiod=period, price=price)[-self._data_num:]

    def get_MACD(self, slow_period, fast_period, signal_period):
        additional_date = slow_period + signal_period
        data = self._get_stock_price_data(additional_date)
        macd, macd_signal, macd_hist = abstract.MACD(data, fast_period, slow_period, signal_period)
        return macd[-self._data_num:]

    def _get_stock_price_data(self, ahead_days):
        new_start_date = get_ahead_date(self.get_start_date(), ahead_days * 2)
        ahead_data = get_all_data_about_stock(self._stock_symbol, start_date=new_start_date,
                                              end_date=self.get_start_date())[:-1]
        ahead_data.extend(self._stock_price)
        remove_date = [i[1:] for i in ahead_data]
        return self._transform_data_to_talib_input(remove_date)

    def _transform_data_to_talib_input(self, data):
        np_array = np.array(data).astype(np.float)
        if self._price_type == self.STOCK_ADJUSTED_CLOSED:
            multiplier = np_array[:, 5] / np_array[:, 3]
        else:
            multiplier = np.ones(np_array.shape[0])
        inputs = {
            self.STOCK_OPEN: np_array[:, 0] * multiplier,
            self.STOCK_HIGH: np_array[:, 1] * multiplier,
            self.STOCK_LOW: np_array[:, 2] * multiplier,
            self.STOCK_CLOSE: np_array[:, 3] * multiplier,
            self.STOCK_VOLUME: np_array[:, 4] * multiplier
        }
        return inputs
