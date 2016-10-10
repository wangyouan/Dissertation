#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: test_data_collection
# Author: Mark Wang
# Date: 10/10/2016

import sys
import logging

from stockforecaster.datacollect import DataCollect
from stockforecaster.constant import Constants

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

const = Constants()

required_info = {
    const.PRICE_TYPE: const.STOCK_CLOSE,
    const.STOCK_PRICE: {const.DATA_PERIOD: 1},
    const.TECHNICAL_INDICATOR: [
        (const.MACD, {const.MACD_FAST_PERIOD: 12, const.MACD_SLOW_PERIOD: 26, const.MACD_TIME_PERIOD: 9}),
        (const.MACD, {const.MACD_FAST_PERIOD: 7, const.MACD_SLOW_PERIOD: 14, const.MACD_TIME_PERIOD: 9}),
        (const.SMA, 3),
        (const.SMA, 13),
        (const.SMA, 21),
        (const.EMA, 5),
        (const.EMA, 13),
        (const.EMA, 21),
        (const.ROC, 13),
        (const.ROC, 21),
        (const.RSI, 9),
        (const.RSI, 14),
        (const.RSI, 21),
    ],
    const.FUNDAMENTAL_ANALYSIS: [
        # const.US10Y_BOND, const.US30Y_BOND,
        # const.FXI,
        # const.IC, const.IA, # comment this  two because this two bond is a little newer
        const.HSI, const.SHSE,
        {const.FROM: const.USD, const.TO: const.HKD},
        {const.FROM: const.EUR, const.TO: const.HKD},
        # {const.FROM: const.AUD, const.TO: const.HKD},
        const.ONE_YEAR, const.HALF_YEAR, const.OVER_NIGHT,
        const.GOLDEN_PRICE,
    ]
}

dc = DataCollect('0001.HK', data_path='/Users/warn/PycharmProjects/Dissertation/data')

df = dc.get_required_data(required_info=required_info, start_date='2012-08-30', end_date='2016-09-30')
df.to_pickle('test.p')
