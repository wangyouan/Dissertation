#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: base_class
# Author: Mark Wang
# Date: 24/7/2016

import os
import logging

import pandas as pd

from stockforecater.constant import Constants


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
            new_df = new_df.drop_duplicates(['Date']).sort(['Date']).reset_index()
            if new_df.shape[0] != df.shape[0]:
                new_df.to_csv(save_path)
        else:
            data_df.to_csv(save_path)

    def _load_data_from_file(self, data_type, file_name):
        if data_type in {self.STOCK_PRICE, self.TECHNICAL_INDICATOR}:
            save_dir = os.path.join(self._data_dir_path, data_type)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        else:
            save_dir = self._data_dir_path

        save_path = os.path.join(save_dir, file_name)
        if os.path.isfile(save_path):
            return pd.read_csv(save_path)
        else:
            return None
