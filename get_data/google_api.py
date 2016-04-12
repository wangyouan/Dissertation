#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: google_api
# Author: Mark Wang
# Date: 12/4/2016
# HKG.1988
# [{u'c': u'-0.06', u'ccol': u'chr', u'e': u'HKG', u'ltt': u'1:30PM GMT+8', u'cp_fix': u'-0.84', u'c_fix': u'-0.06', u'l': u'7.06', u's': u'0', u'lt': u'Apr 12, 1:30PM GMT+8', u'pcls_fix': u'7.12', u't': u'1988', u'lt_dts': u'2016-04-12T13:30:16Z', u'l_fix': u'7.06', u'cp': u'-0.84', u'id': u'15757850', u'l_cur': u'HK$7.06'}]


import urllib2  # works fine with Python 2.7.9 (not 3.4.+)
import json
import time


def fetch_real_time_info(symbol, exchange):
    link = "http://finance.google.com/finance/info?client=ig&q="
    url = "{}{}:{}".format(link, exchange, symbol)
    u = urllib2.urlopen(url)
    content = u.read()
    data = json.loads(content[3:])
    info = data[0]
    print data
    t = str(info["elt"])  # time stamp
    l = float(info["l"])  # close price (previous trading day)
    p = float(info["el"])  # stock price in pre-market (after-hours)
    return (t, l, p)


p0 = 0
while True:
    t, l, p = fetch_real_time_info("1988", "HKG")
    if (p != p0):
        p0 = p
        print("%s\t%.2f\t%.2f\t%+.2f\t%+.2f%%" % (t, l, p, p - l,
                                                  (p / l - 1) * 100.))
    time.sleep(60)
