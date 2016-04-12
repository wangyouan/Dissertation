#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: yahoo_api.py
# Author: Mark Wang
# Date: 12/4/2016

from urllib2 import Request, urlopen
from urllib import urlencode


def get_all_date_about_stock(symbol):
    url = "http://ichart.finance.yahoo.com/table.csv?s={}".format(symbol)
    query = Request(url)
    response = urlopen(query)
    f = open(r'../data/{}.csv'.format(symbol), 'w')
    f.write(response.read())
    f.close()


if __name__ == "__main__":
    get_all_date_about_stock("1988.HK")
