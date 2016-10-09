#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: get_hibor
# Author: Mark Wang
# Date: 12/6/2016

import datetime

import numpy as np
from BeautifulSoup import BeautifulSoup

from stockforecaster.util.date_util import str2datetime
from stockforecaster.util.http_method import get


def get_hk_interest_rate(detail_date):
    url = "http://www.hkab.org.hk/hibor/listRates.do"
    if not isinstance(detail_date, datetime.datetime):
        detail_date = str2datetime(detail_date)
    data_list = [('lang', 'en'), ('Submit', 'Search'), ('year', detail_date.year), ('month', detail_date.month),
                 ('day', detail_date.day)]
    page_html = get(url, data_list)
    soup = BeautifulSoup(page_html)
    rate_info = {"Overnight": np.nan,
                 "1 Week": np.nan,
                 # "2 Weeks": np.nan,
                 "1 Month": np.nan,
                 "2 Months": np.nan,
                 "3 Months": np.nan,
                 # "4 Months": np.nan,
                 # "5 Months": np.nan,
                 "6 Months": np.nan,
                 # "7 Months": np.nan,
                 # "8 Months": np.nan,
                 # "9 Months": np.nan,
                 # "10 Months": np.nan,
                 # "11 Months": np.nan,
                 "12 Months": np.nan,
                 }
    for table in soup('table'):
        if table.get('class') == 'etxtmed' and table.get('bgcolor') == '#ffffff':
            break

    else:
        return None

    td_list = table('td')
    for i in range(len(td_list)):
        if td_list[i].text in rate_info:
            rate_info[td_list[i].text] = float(td_list[i + 1].text)

    return rate_info

if __name__ == '__main__':
    print get_hk_interest_rate('2016-09-20')
