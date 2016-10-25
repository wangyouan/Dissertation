#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: base_class
# Author: Mark Wang
# Date: 24/7/2016

import os
import logging
import pickle

import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday

from stockforecaster.constant import Constants
from stockforecaster.util.date_util import *
from stockforecaster.util.query_information import get_hk_interest_rate, get_yahoo_finance_data, query_quandl_data


class BaseClass(Constants):
    def __init__(self, stock_symbol, logger=None, data_dir_path=None):
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getLogger(self.__class__.__name__)

        self._data_dir_path = data_dir_path
        if not os.path.isdir(self._data_dir_path):
            os.makedirs(self._data_dir_path)
        self._stock_symbol = stock_symbol
        self._start_date = None
        self._end_date = None
        self._is_adjusted = False
        self._required_date_list = None

        # bond ticker to query
        self._bond_label_dict_us = {
            self.US10Y_BOND: "^TNX",
            self.US30Y_BOND: "^TYX",
            self.FXI: "FXI",
        }
        self._bond_label_dict_hk = {
            self.HSI: "^HSI",
            self.IC: "2801.HK",
            self.IA: "2829.HK",
            self.IA300: "2846.HK",
            self.IMSCI: "3010.HK",
            self.SHSE: '000001.SS',
        }

    def _save_data_to_file(self, data_df, file_name, data_type):
        if data_type in {self.STOCK_PRICE, self.TECHNICAL_INDICATOR}:
            save_dir = os.path.join(self._data_dir_path, data_type)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        else:
            save_dir = self._data_dir_path

        save_path = os.path.join(save_dir, file_name)

        if os.path.isfile(save_path):
            previous_file = pd.read_pickle(save_path)
            data_df = pd.concat([previous_file, data_df], axis=0)
            data_df = data_df.groupby(data_df.index).first()
        data_df.to_pickle(save_path)

    def _load_data_from_file(self, data_type, file_name):
        if data_type in {self.STOCK_PRICE, self.TECHNICAL_INDICATOR}:
            save_dir = os.path.join(self._data_dir_path, data_type)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        else:
            save_dir = self._data_dir_path

        save_path = os.path.join(save_dir, file_name)
        if os.path.isfile(save_path):
            if data_type == self.HONG_KONG:
                with open(save_path) as f:
                    return pickle.load(f)
            else:
                return pd.read_pickle(save_path)
        else:
            if data_type == self.STOCK_PRICE:
                self.logger.warn('No previous file founded or previous file data is not enough, '
                                 'will load from yahoo finance')
                data_df = get_yahoo_finance_data(self._stock_symbol, remove_zero_volume=True)
                self._save_data_to_file(data_df, self._stock_symbol, self.STOCK_PRICE)
                return data_df

            else:
                return pd.DataFrame()

    def set_start_date(self, detail_date):
        if isinstance(detail_date, str):
            detail_date = str2datetime(detail_date)

        self._start_date = detail_date

    def set_end_date(self, detail_date):
        if isinstance(detail_date, str):
            detail_date = str2datetime(detail_date)

        self._end_date = detail_date

    def get_start_date(self):
        return self._start_date

    def get_end_date(self):
        return self._end_date

    def set_is_adjusted(self, is_adjusted):
        self._is_adjusted = is_adjusted

    def get_is_adjusted(self):
        return self._is_adjusted

    def check_fundamental_data_integrity(self, data_df, data_type, curr_from=None, curr_to=None):
        self.logger.info('Start to check the integrity of target data type {}'.format(data_type))
        set_current = set(data_df.index)
        set_require = set(self._required_date_list)
        diff = set_require.difference(set_current)
        if not set_current or min(set_current) > min(set_require) or max(set_current) < max(set_require):

            self.logger.info('Data need to be updated')
            self.logger.debug('Missed Data is {}'.format(diff))
            self.logger.debug('Start date is {}, end date is {}'.format(min(diff), max(diff)))
            if data_type == self.HIBOR:
                file_name = self.HIBOR
                if data_df.empty:
                    data_df = pd.DataFrame(
                        columns=['12 Months', '3 Months', 'Overnight', '1 Week', '6 Months', '1 Month', '2 Months'])
                for miss_date in diff:
                    data_df.loc[miss_date] = get_hk_interest_rate(miss_date)

            elif data_type == self.CURRENCY_EXCHANGE:
                file_name = '{}2{}'.format(curr_from, curr_to)
                file_name_dict = {"{}2{}".format(self.EUR, self.HKD): 'BOE/XUDLHDS',
                                  "{}2{}".format(self.USD, self.HKD): 'BOE/XUDLHDD',
                                  "{}2{}".format(self.AUD, self.HKD): 'RBA/FXRHKD',
                                  }
                miss_df = query_quandl_data(file_name_dict[file_name], min(diff), max(diff))
                data_df = pd.concat([data_df, miss_df], axis=0)
                data_df = data_df.groupby(data_df.index).first()

            elif data_type == self.BOND:

                file_name = curr_to

                # this branch for ordinary info
                if curr_from in self._bond_label_dict_hk:
                    miss_df = get_yahoo_finance_data(self._bond_label_dict_hk[curr_from], min(diff), max(diff), False)

                # this branch for us info
                else:
                    miss_df = get_yahoo_finance_data(self._bond_label_dict_us[curr_from], min(diff), max(diff), False)
                    hk_cal_rules = self._load_data_from_file(self.HONG_KONG, 'hk_cal_rules.p')
                    business_day = CustomBusinessDay(
                        calendar=AbstractHolidayCalendar(self.HONG_KONG, rules=hk_cal_rules))
                    miss_df['Date'] = miss_df.index
                    miss_df['Date'] = miss_df['Date'].apply(lambda x: x + business_day)
                    miss_df.set_index('Date', inplace=True)

                # data_df = pd.concat([data_df, miss_df], axis=0).drop_duplicates().sort_index()
                data_df = pd.concat([data_df, miss_df], axis=0)
                data_df = data_df.groupby(data_df.index).first()

            elif data_type == self.GOLDEN_PRICE:
                file_name = self.TYPE_GOLDEN_PRICE
                miss_df = query_quandl_data('WGC/GOLD_DAILY_CNY', min(diff), max(diff))
                # data_df = pd.concat([data_df, miss_df], axis=0).drop_duplicates().sort_index()
                data_df = pd.concat([data_df, miss_df], axis=0)
                data_df = data_df.groupby(data_df.index).first()

            else:
                self.logger.warn('Unknown data type {}'.format(data_type))
                raise Exception('Unknown data type {}'.format(data_type))

            self._save_data_to_file(data_df, '{}.p'.format(file_name), data_type)

        self.logger.info('Finish check data integrity')
        return data_df
