#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: test_current_code
# Author: Mark Wang
# Date: 14/10/2016

from stockforecaster import StockForecaster as SF

stock_symbol = '0001.HK'
start_date = '2012-08-10'
end_date = '2016-08-10'
test_date = '2015-08-10'
data_path = '/Users/warn/PycharmProjects/Dissertation/data'

if __name__ == '__main__':
    sf = SF(stock_symbol=stock_symbol, data_path=data_path,
            train_method=SF.RANDOM_FOREST,
            train_system=SF.SPARK)

    result = sf.main_process(start_date=start_date, end_date=end_date, test_start_date=test_date)
    result.to_csv('test_result.csv')
