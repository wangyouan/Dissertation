#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: query_yahoo_finance
# Author: Mark Wang
# Date: 24/7/2016

import datetime
from StringIO import StringIO

import pandas as pd

from stockforecaster.util.http_method import get


def get_yahoo_finance_data(symbol, start_date=None, end_date=None, remove_zero_volume=True):
    """
    Using yahoo finance API Get stock price with high low open close data

    :param symbol: stock symbol used in yahoo finance
    :param start_date: start date of the given stock data 2012-03-15
    :param end_date: end data
    :param remove_zero_volume: if True, will remove all data with zero volume
    :return: a list of stock price as [date, open, high, low, close]
    """
    data_list = [('s', symbol)]
    if start_date:
        data = start_date.split('-')
        data_list.append(('a', int(data[1]) - 1))
        data_list.append(('b', data[2]))
        data_list.append(('c', data[0]))
    if end_date:
        data = end_date.split('-')
        data_list.append(('d', int(data[1]) - 1))
        data_list.append(('e', data[2]))
        data_list.append(('f', data[0]))
    data_list.append(('g', 'd'))
    data_list.append(('ignore', '.csv'))

    url = "http://chart.finance.yahoo.com/table.csv"
    stock_info = get(url=url, data_list=data_list)
    stock_data = StringIO(stock_info)
    stock_df = pd.read_csv(stock_data)
    stock_df['Date'] = stock_df['Date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    stock_df = stock_df.sort_values(by=['Date'])

    if not remove_zero_volume:
        return stock_df

    return stock_df[stock_df['Volume'] > 0].reset_index(drop=True)

if __name__ == '__main__':
    df = get_yahoo_finance_data('0001.HK')
    df.to_pickle('0001.HK')
