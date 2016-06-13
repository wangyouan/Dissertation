#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: predict_price
# Author: Mark Wang
# Date: 13/6/2016

import os
import sys

from StockInference.inference_system import InferenceSystem
from StockInference.constant import Constants

const = Constants()

data_path = 'data'
symbol = "0001.HK"
features = {
    const.PRICE_TYPE: const.STOCK_CLOSE,
    const.STOCK_PRICE: {const.DATA_PERIOD: 5},
    const.STOCK_INDICATOR: [
        (const.MACD, {
            const.MACD_FAST_PERIOD: 12,
            const.MACD_SLOW_PERIOD: 26,
            const.MACD_TIME_PERIOD: 9
        }),
        (const.MACD, {
            const.MACD_FAST_PERIOD: 7,
            const.MACD_SLOW_PERIOD: 14,
            const.MACD_TIME_PERIOD: 9
        }),
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
        const.US10Y_BOND,
        const.US30Y_BOND,
        const.FXI,
        const.IC,
        const.IA,  # comment this  two because this two bond is a little newer
        const.HSI,
        {const.FROM: const.USD, const.TO: const.HKD},
        {const.FROM: const.EUR, const.TO: const.HKD},
        {const.FROM: const.AUD, const.TO: const.HKD},
        const.ONE_YEAR,
        const.HALF_YEAR,
        const.OVER_NIGHT,
        const.GOLDEN_PRICE,
    ]
}

if sys.platform == 'darwin':
    data_path = os.path.join('..', data_path)

test = InferenceSystem(stock_symbol=symbol)
date, price = test.get_future_stock_price(const.ARTIFICIAL_NEURAL_NETWORK, data_file_path=data_path, features=features)
print "The price of", symbol, "in", date, 'is', price
