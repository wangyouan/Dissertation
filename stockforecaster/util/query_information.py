#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: get_hibor
# Author: Mark Wang
# Date: 12/6/2016

import datetime
from StringIO import StringIO

import numpy as np
import pandas as pd
import quandl
from BeautifulSoup import BeautifulSoup

from stockforecaster.util.date_util import str2datetime
from stockforecaster.util.http_method import get

quandl.ApiConfig.api_key = "RYdPmBZoFyLXxg1RQ3fY"


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


def get_yahoo_finance_data(symbol, start_date=None, end_date=None, remove_zero_volume=True):
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
    data_list.append(('ignore', '.csv'))

    url = "http://chart.finance.yahoo.com/table.csv"
    stock_info = get(url=url, data_list=data_list)
    stock_data = StringIO(stock_info)
    stock_df = pd.read_csv(stock_data)
    stock_df['Date'] = stock_df['Date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    stock_df = stock_df.set_index('Date').sort_index()

    if remove_zero_volume:
        return stock_df[stock_df['Volume'] > 0]
    else:
        return stock_df


def query_quandl_data(query_info, start_date=None, end_date=None, transform=None):
    """
    query info from Quandl

    :param query_info: target info to query
    :param start_date: start date
    :param end_date: end date
    :param transform: The following are useful transform types
        none	no effect	y"[t] = y[t]
        diff	row-on-row change	y"[t] = y[t] – y[t-1]
        rdiff	row-on-row % change	y"[t] = (y[t] – y[t-1]) / y[t-1]
        rdiff_from	latest value as % increment	y"[t] = (y[latest] – y[t]) / y[t]
        cumul	cumulative sum	y"[t] = y[0] + y[1] + … + y[t]
        normalize	scale series to start at 100	y"[t] = y[t] ÷ y[0] * 100
    :return: query result
    """
    data = quandl.get(query_info, start_date=start_date, end_date=end_date, returns='pandas', transform=transform)
    return data


if __name__ == '__main__':
    data_df = query_quandl_data('RBA/FXRHKD')
    print data_df.keys()
    print query_quandl_data('FRED/DEXHKUS').keys()
    print query_quandl_data('ECB/EURHKD').keys()
