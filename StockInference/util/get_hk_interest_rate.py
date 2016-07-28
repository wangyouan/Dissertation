#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: get_hk_interest_rate
# Author: Mark Wang
# Date: 12/6/2016

from BeautifulSoup import BeautifulSoup

from StockInference.util.date_parser import string_to_datetime
from stockforecaster.util.http_get import get


def get_hk_interest_rate(detail_date):
    url = "http://www.hkab.org.hk/hibor/listRates.do"
    detail_date = string_to_datetime(detail_date)
    data_list = [('lang', 'en'), ('Submit', 'Search'), ('year', detail_date.year), ('month', detail_date.month),
                 ('day', detail_date.day)]
    page_html = get(url, data_list)
    soup = BeautifulSoup(page_html)
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
