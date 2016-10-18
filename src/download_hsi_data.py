#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: download_hsi_data
# Author: Mark Wang
# Date: 18/10/2016

import os
import time

from stockforecaster.util.query_information import get_yahoo_finance_data

stock_list = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK', '0006.HK', '0011.HK', '0012.HK', '0016.HK',
              '0017.HK', '0019.HK', '0023.HK', '0027.HK', '0066.HK', '0083.HK', '0101.HK', '0135.HK', '0144.HK',
              '0151.HK', '0267.HK', '0293.HK', '0386.HK', '0388.HK', '0494.HK', '0688.HK', '0700.HK', '0762.HK',
              '0823.HK', '0836.HK', '0857.HK', '0883.HK', '0939.HK', '0941.HK', '0992.HK', '1038.HK', '1044.HK',
              '1088.HK', '1109.HK', '1113.HK', '1299.HK', '1398.HK', '1880.HK', '1928.HK', '2018.HK', '2318.HK',
              '2319.HK', '2388.HK', '2628.HK', '3328.HK', '3988.HK']

output_path = '/Users/warn/PycharmProjects/Dissertation/data/stock_price'

for stock in stock_list:
    df = get_yahoo_finance_data(stock)
    print min(df.index), max(df.index)
    df.to_pickle(os.path.join(output_path, stock))
    time.sleep(1)
