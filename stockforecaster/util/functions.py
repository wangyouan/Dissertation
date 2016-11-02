#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: functions
# Author: Mark Wang
# Date: 20/10/2016


import pandas as pd

from dateutil.relativedelta import relativedelta

from stockforecaster.stock_forecaster import StockForecaster as SF
from stockforecaster.util.date_util import str2datetime


def predict_stock_price_spark(stock_symbol, data_path, train_method, start_date, end_date, test_date,
                              using_percentage=True, window_size=None, worker_num=2, hidden_nodes_num=None,
                              rt_trees_num=30):
    """
    given required stock type, return its prediction result.
    :param stock_symbol: target stock symbol
    :param data_path: data saved path
    :param train_method: training method
    :param start_date: total start date
    :param end_date: total end date
    :param test_date: test info start date
    :param using_percentage: if training method is combine system, this parameter are used to choose which price is
    used
    :param window_size: the month number of target input, if None, will not do data separation
    :param worker_num: number of spark executor numbers
    :param hidden_nodes_num: the number of hidden layer nodes in parameter
    :param rt_trees_num: the number of random forest trees number
    :return: prediction result
    """
    sf = SF(stock_symbol=stock_symbol, data_path=data_path,
            train_method=train_method, worker_num=worker_num,
            train_system=SF.SPARK, using_percentage=using_percentage,
            rt_trees_num=rt_trees_num, ann_hidden_nodes=hidden_nodes_num)

    if window_size is None:
        result = sf.main_process(start_date=start_date, end_date=end_date, test_start_date=test_date)
    else:
        start_date = str2datetime(start_date)
        end_date = str2datetime(end_date)
        test_date = str2datetime(test_date)

        month_dict = {SF.ONE_MONTH: 1,
                      SF.TWO_MONTHS: 2,
                      SF.THREE_MONTHS: 3,
                      SF.HALF_YEAR: 6}
        if not isinstance(window_size, int):
            if window_size not in month_dict:
                raise ValueError('Unknown window_size {}'.format(window_size))
            else:
                window_size = month_dict[window_size]

        delta_month = relativedelta(months=window_size)
        tmp_end_time = test_date + delta_month

        dfs = []

        while test_date < end_date:
            print 'Start time:', start_date
            if tmp_end_time > end_date:
                tmp_end_time = end_date
            dfs.append(sf.main_process(start_date=start_date, end_date=tmp_end_time, test_start_date=test_date))
            # time.sleep(10)
            start_date += delta_month
            tmp_end_time += delta_month
            test_date += delta_month

        result = pd.concat(dfs, axis=0).sort_index()

    sf.stop_server()
    return result
