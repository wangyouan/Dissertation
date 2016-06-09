#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: http_get
# Author: Mark Wang
# Date: 8/6/2016

import time
import urllib
import urllib2

from BeautifulSoup import BeautifulSoup


def get(url, data_list=None, timeout=10):

    if data_list:
        url = "{}?{}".format(url, urllib.urlencode(data_list))
    query = urllib2.Request(url)
    max_try = 3
    current_try = 0
    while current_try < max_try:
        try:
            response = urllib2.urlopen(query, timeout=timeout)
            html = response.read()
            response.close()
            return html
        except Exception, e:
            print e
            current_try += 1
            time.sleep(timeout)
    raise Exception("Cannot open page {}".format(url))


def get_currency_exchange_rate(from_currency, date):
    url = "http://www.x-rates.com/historical/"
    data_list = [('from', from_currency), ('amount', 1), ('date', date)]
    page = get(url, data_list)
    print page
    soup = BeautifulSoup(page)
    f = open('test.html', 'w')
    f.write(page)
    f.close()


if __name__ == "__main__":
    get_currency_exchange_rate('HKD', '2006-05-25')
