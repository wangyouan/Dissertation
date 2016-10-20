#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: prepare_train_test_data
# Author: Mark Wang
# Date: 11/10/2016

from stockforecaster import StockForecaster


sf = StockForecaster(stock_symbol='0001.HK', data_path='/Users/warn/PycharmProjects/Dissertation/data',
                     train_method=StockForecaster.LINEAR_REGRESSION,
                     train_system=StockForecaster.SPARK)
sf.main_process('2010-08-10', '2016-08-10', '2015-08-10')
