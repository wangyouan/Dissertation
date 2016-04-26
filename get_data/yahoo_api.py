#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: yahoo_api.py
# Author: Mark Wang
# Date: 12/4/2016

from urllib2 import Request, urlopen
from urllib import urlencode


def get_all_date_about_stock(symbol, start_date=None, end_date=None):
    date_list = [('s', symbol)]
    if start_date:
        date = start_date.split('-')
        date_list.append(('a', date[1]))
        date_list.append(('b', date[2]))
        date_list.append(('c', date[0]))
    if end_date:
        date = end_date.split('-')
        date_list.append(('d', date[1]))
        date_list.append(('e', date[2]))
        date_list.append(('f', date[0]))
    date_list.append(('g', 'd'))

    url = "http://ichart.finance.yahoo.com/table.csv?{}".format(urlencode(date_list))
    query = Request(url)
    response = urlopen(query)
    f = open(r'../data/{}.csv'.format(symbol), 'w')
    f.write(response.read())
    f.close()


if __name__ == "__main__":
    # stock_symbol = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK']
    stock_symbol = []
    for i in range(50, 66):
        stock_symbol.append("{:04d}.HK".format(i))
    # print stock_symbol
    for symbol in stock_symbol:
        try:
            get_all_date_about_stock(symbol, start_date='2006-03-14', end_date='2016-03-15')
        except Exception:
            print("Get stock {} failed".format(symbol))
