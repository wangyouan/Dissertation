#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: get_hongkong_interest
# Author: Mark Wang
# Date: 11/6/2016

import datetime
import urllib

import BeautifulSoup
from pandas.tseries.offsets import CustomBusinessDay

from StockInference.util.hongkong_calendar import HongKongCalendar
from stockforecaster.util.http_get import get

url = "http://www.hkab.org.hk/hibor/listRates.do?"


def get_interest_rate():
    start_datetime = datetime.datetime(2006, 1, 1)
    custom_day = CustomBusinessDay(calendar=HongKongCalendar(2006, 2016))
    start_datetime += custom_day
    rate_dict = {}
    while start_datetime < datetime.datetime.today():
        rate_dict[start_datetime.strftime("%Y-%m-%d")] = get_someday_rate(start_datetime)
        start_datetime += custom_day

    import pickle
    f = open("interest_rate.dat", 'w')
    pickle.dump(rate_dict, f)
    f.close()

    import pprint
    pprint.pprint(rate_dict, width=120)


def get_someday_rate(detail_date):
    data_list = [('lang', 'en'), ('Submit', 'Search'), ('year', detail_date.year), ('month', detail_date.month),
                 ('day', detail_date.day)]
    new_url = '{}{}'.format(url, urllib.urlencode(data_list))
    page_html = get(new_url)
    soup = BeautifulSoup.BeautifulSoup(page_html)
    rate_info = {"Overnight": None,
                 "1 Week": None,
                 "2 Weeks": None,
                 "1 Month": None,
                 "2 Months": None,
                 "3 Months": None,
                 "4 Months": None,
                 "5 Months": None,
                 "6 Months": None,
                 "7 Months": None,
                 "8 Months": None,
                 "9 Months": None,
                 "10 Months": None,
                 "11 Months": None,
                 "12 Months": None,
                 }
    for table in soup('table'):
        if table.get('class') == 'etxtmed' and table.get('bgcolor') == '#ffffff':
            break

    else:
        return None

    td_list = table('td')
    for i in range(len(td_list)):
        if td_list[i].text in rate_info:
            rate_info[td_list[i].text] = td_list[i + 1].text

    return rate_info


if __name__ == "__main__":
    get_interest_rate()
