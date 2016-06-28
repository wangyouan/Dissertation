#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: get_history_stock_price
# Author: Mark Wang
# Date: 29/5/2016

import time
from urllib import urlencode
from urllib2 import Request, urlopen, URLError, HTTPError


def get_all_data_about_stock(symbol, start_date=None, end_date=None, remove_zero_volume=True):
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

    url = "http://ichart.finance.yahoo.com/table.csv?{}".format(urlencode(data_list))
    query = Request(url)
    max_try = 3
    current_try = 0
    while current_try < max_try:
        try:
            response = urlopen(query)
            stock_info = response.read()
            stock_info = [i.split(',') for i in stock_info.split('\n')][1:-1]
            stock_info = [[i[0], float(i[1]), float(i[2]), float(i[3]), float(i[4]), int(i[5]),
                           float(i[6])] for i in stock_info]
            stock_info.reverse()
            if not remove_zero_volume:
                return stock_info
            stock_list = []
            for info in stock_info:
                if info[5] == 0:
                    continue
                else:
                    stock_list.append(info)
            return stock_list
        except HTTPError, e:
            print e
            print "No data could be found"
            return []
        except Exception, e:
            print e
            current_try += 1
            time.sleep(10)
    raise Exception("Cannot open page {}".format(url))


if __name__ == "__main__":
    a = get_all_data_about_stock("0003.HK", start_date='2012-01-06', end_date='2014-01-05')
    print 247 * 5
    print len(a)
