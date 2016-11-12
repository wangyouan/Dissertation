#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: get_avg_price
# Author: Mark Wang
# Date: 11/11/2016

import os

import pandas as pd


def add_average_price(path):
    statistic = pd.read_csv(os.path.join(path, 'statistics.csv'), index_col=0)
    statistic.loc[:, 'avgPrice'] = 0
    for symbol in statistic.index:
        if symbol == '^HSI':
            df = pd.read_csv(os.path.join(path, '^HSI.csv'), index_col=0)
        else:
            df = pd.read_csv(os.path.join(path, symbol.replace('HK', 'csv')), index_col=0)

        statistic.loc[symbol, 'avgPrice'] = df['Target'].mean()

    statistic.to_csv(os.path.join(path, 'statisticsAddPrice.csv'))


if __name__ == '__main__':
    for x, y, z in os.walk('/Users/warn/Documents/MScDissertation/Data/DataFrame/start_date/2013-07-06'):
        if 'statistics.csv' in z:
            add_average_price(x)

