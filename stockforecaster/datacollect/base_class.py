#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: base_class
# Author: Mark Wang
# Date: 24/7/2016

import os
import logging

import pandas as pd

from stockforecaster.constant import Constants
from stockforecaster.util.date_util import *


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

    def _save_data_to_file(self, data_df, file_name, data_type):
        if data_type in {self.STOCK_PRICE, self.TECHNICAL_INDICATOR}:
            save_dir = os.path.join(self._data_dir_path, data_type)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        else:
            save_dir = self._data_dir_path

        save_path = os.path.join(save_dir, file_name)
        if os.path.isfile(save_path):
            df = pd.read_csv(save_path)
            new_df = pd.concat([data_df, df], axis=0, ignore_index=True)
            new_df = new_df.drop_duplicates(['Date']).sort(['Date']).reset_index(drop=True)
            if new_df.shape[0] != df.shape[0]:
                new_df.to_pickle(save_path)
        else:
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
            return pd.read_pickle(save_path)
        else:
            return None

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