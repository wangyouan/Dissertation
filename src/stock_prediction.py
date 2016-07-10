#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: stock_prediction
# Author: Mark Wang
# Date: 13/6/2016

import os
import sys
import time
import traceback

from StockInference.inference_system import InferenceSystem
from StockInference.util.data_parse import *
from StockInference.constant import Constants
from __init__ import start_date, end_date, test_ratio, predict_list

const = Constants()
test_times = 10

if len(sys.argv) > 3:
    date_start = sys.argv[1]
    date_end = sys.argv[2]
    ratio = sys.argv[3]
else:
    date_start = start_date
    date_end = end_date
    ratio = test_ratio

required_info = {
    const.PRICE_TYPE: const.STOCK_ADJUSTED_CLOSED,
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
        # const.US10Y_BOND,
        # const.US30Y_BOND,
        const.FXI,
        const.IC,
        const.IA,  # comment this  two because this two bond is a little newer
        const.HSI,
        {const.FROM: const.USD, const.TO: const.HKD},
        {const.FROM: const.EUR, const.TO: const.HKD},
        # {const.FROM: const.AUD, const.TO: const.HKD},
        const.ONE_YEAR,
        const.HALF_YEAR,
        # const.OVER_NIGHT,
        # const.GOLDEN_PRICE,
        const.SHSE,
    ]
}

output_path = 'output'
data_path = 'data'
model_path = "models"

if sys.platform == 'darwin':
    output_path = '../{}'.format(output_path)
    data_path = '../{}'.format(data_path)
    model_path = '../{}'.format(model_path)

if not os.path.isdir(data_path):
    os.makedirs(data_path)
stock_list = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK', '0006.HK', '0011.HK', '0012.HK', '0014.HK',
              '0016.HK', '0017.HK', '0019.HK', '0023.HK', '0027.HK', '0031.HK', '0033.HK', '0043.HK', '0064.HK',
              '0066.HK', '0069.HK', '0076.HK', '0078.HK', '0083.HK', '0101.HK', '0116.HK', '0119.HK', '0121.HK',
              '0123.HK', '0144.HK', '0148.HK', '0151.HK', '0152.HK', '0268.HK', '0291.HK', '0455.HK', '0546.HK',
              '0688.HK', '0700.HK', '0737.HK', '0777.HK', '0845.HK', '1051.HK', '1112.HK', '1117.HK', '1361.HK',
              '1918.HK', '2005.HK', '2362.HK', '2383.HK', '6823.HK']

test = None
for method in [const.RANDOM_FOREST, const.LINEAR_REGRESSION, const.ARTIFICIAL_NEURAL_NETWORK][2:]:

    new_file_path = os.path.join(output_path, method.lower())
    if not os.path.isdir(new_file_path):
        os.makedirs(new_file_path)

    f = open(os.path.join(new_file_path, "stock_info.csv"), 'w')
    f.write('stock,MSE,MAPE,MAD,RMSE,CDC,HMSE,ME\n')
    # for stock in stock_list[:10]:
    for stock in predict_list[:1]:
        me = 0.0
        mse = 0.0
        mape = 0.0
        mad = 0.0
        rmse = 0.0
        hmse = 0.0
        cdc = 0.0

        specific_file_path = os.path.join(new_file_path, stock[:4])
        specific_model_path = os.path.join(model_path, method, stock[:4])
        for i in range(test_times):
            test = InferenceSystem(stock, training_method=method, data_folder_path=data_path, using_exist_model=False,
                                   output_file_path=specific_file_path, model_path=specific_model_path)
            try:
                predict_result = test.predict_historical_data(ratio, date_start, date_end, iterations=10)
                predict_result.cache()
                me += get_ME(predict_result)
                mse += get_MSE(predict_result)
                mape += get_MAPE(predict_result)
                mad += get_MAD(predict_result)
                rmse += get_RMSE(predict_result)
                hmse += get_HMSE(predict_result)
                # tie = get_theils_inequality_coefficient(predict_result)
                cdc += get_CDC_combine(predict_result)
            except Exception, err:
                traceback.print_exc()
                time.sleep(20)

        f.write('{},{},{},{},{},{},{},{}\n'.format(stock, mse / test_times, mape / test_times, mad / test_times,
                                                   rmse / test_times, cdc / test_times, hmse / test_times,
                                                   me / test_times))

    f.close()

    if hasattr(test, 'sc'):
        test.sc.stop()
        time.sleep(60)
