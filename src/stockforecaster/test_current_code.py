#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: test_current_code
# Author: Mark Wang
# Date: 14/10/2016

from stockforecaster import StockForecaster as SF
from stockforecaster.util.evaluate_func import calculate_mean_squared_error, \
    calculate_success_direction_prediction_rate, calculate_mean_absolute_percentage_error

stock_symbol = '0018.HK'
start_date = '2013-01-06'
end_date = '2016-01-06'
test_date = '2015-01-06'
data_path = '/Users/warn/PycharmProjects/Dissertation/data'

if __name__ == '__main__':
    sf = SF(stock_symbol=stock_symbol, data_path=data_path,
            # train_method={
            #     SF.CHANGE_DIRECTION: SF.RANDOM_FOREST,
            #     SF.CHANGE_AMOUNT: SF.ARTIFICIAL_NEURAL_NETWORK
            # },
            train_method=SF.RANDOM_FOREST,
            train_system=SF.SPARK, using_percentage=True)

    result = sf.main_process(start_date=start_date, end_date=end_date, test_start_date=test_date)
    result.to_csv('test_result.csv')

    print calculate_success_direction_prediction_rate(result, SF.TODAY_PRICE, 'prediction', SF.TARGET_PRICE) * 100
    print calculate_mean_squared_error(result, 'prediction', SF.TARGET_PRICE)
    print calculate_mean_absolute_percentage_error(result, 'prediction', SF.TARGET_PRICE)
