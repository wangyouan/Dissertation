#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: get_history_stock_price
# Author: Mark Wang
# Date: 29/5/2016

from urllib import urlencode
from urllib2 import Request, urlopen


def get_all_data_about_stock(symbol, start_date=None, end_date=None):
    """
    Using yahoo finance API Get stock price with high low open close data

    :param symbol: stock symbol used in yahoo finance
    :param start_date: start date of the given stock data 2012-03-15
    :param end_date: end data
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

    url = "http://ichart.finance.yahoo.com/table.csv?{}".format(urlencode(data_list))
    query = Request(url)
    response = urlopen(query)
    stock_info = response.read()
    stock_info = [i.split(',') for i in stock_info.split('\n')][1:-1]
    stock_info = [i[:5] for i in stock_info]
    stock_info.reverse()
    return stock_info


if __name__ == "__main__":
    print get_all_data_about_stock("0003.HK", start_date='2016-02-14', end_date='2016-03-15')
