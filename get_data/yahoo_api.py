#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: yahoo_api.py
# Author: Mark Wang
# Date: 12/4/2016

import os
from urllib2 import Request, urlopen
from urllib import urlencode


def get_all_date_about_stock(symbol, start_date=None, end_date=None, save_path='../data'):
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

    url = "http://chart.finance.yahoo.com/table.csv?{}".format(urlencode(date_list))
    query = Request(url)
    response = urlopen(query)
    f = open(os.path.join(save_path, '{}.csv'.format(symbol)), 'w')
    f.write(response.read())
    f.close()


if __name__ == "__main__":
    # stock_symbol = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK']
    hsi_stock_symbol = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK', '0006.HK', '0011.HK', '0012.HK',
                        '0016.HK', '0017.HK', '0019.HK', '0023.HK', '0027.HK', '0066.HK', '0083.HK', '0101.HK',
                        '0135.HK', '0144.HK', '0151.HK', '0267.HK', '0293.HK', '0322.HK', '0386.HK', '0388.HK',
                        '0494.HK', '0688.HK', '0700.HK', '0762.HK', '0823.HK', '0836.HK', '0857.HK', '0883.HK',
                        '0939.HK', '0941.HK', '0992.HK', '1038.HK', '1044.HK', '1088.HK', '1109.HK', '1113.HK',
                        '1299.HK', '1398.HK', '1880.HK', '1928.HK', '2318.HK', '2319.HK', '2388.HK', '2628.HK',
                        '3328.HK', '3988.HK']

    import sys

    if len(sys.argv) > 1:
        save_path = sys.argv[1]
    else:
        save_path = '../data'
    # stock_symbol = []
    # for i in range(68, 69):
    #     stock_symbol.append("{:04d}.HK".format(i))
    # print stock_symbol
    for symbol in hsi_stock_symbol[:1]:
        try:
            get_all_date_about_stock(symbol, save_path=save_path)
        except Exception:
            import traceback

            traceback.print_exc()
            print("Get stock {} failed".format(symbol))
