#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: get_history_stock_price
# Author: Mark Wang
# Date: 29/5/2016

from urllib import urlencode
from urllib2 import Request, urlopen

def get_all_data_about_stock(symbol, start_date=None, end_date=None):
    data_list = [('s', symbol)]
    if start_date:
        data = start_date.split('-')
        data_list.append(('a', data[1]))
        data_list.append(('b', data[2]))
        data_list.append(('c', data[0]))
    if end_date:
        data = end_date.split('-')
        data_list.append(('d', data[1]))
        data_list.append(('e', data[2]))
        data_list.append(('f', data[0]))
    data_list.append(('g', 'd'))

    url = "http://ichart.finance.yahoo.com/table.csv?{}".format(urlencode(data_list))
    query = Request(url)
    response = urlopen(query)
    stock_info =  response.read()
    stock_info = [i.split(',') for i in stock_info.split('\n')][1:-1]
    return stock_info