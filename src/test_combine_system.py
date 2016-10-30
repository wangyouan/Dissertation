#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: test_combine_system
# Author: Mark Wang
# Date: 18/6/2016


import os
import sys
import time
import traceback

from StockInference.composition_prediction_system import MixInferenceSystem
from StockInference.util.data_parse import *
from StockInference.constant import Constants
from __init__ import start_date, end_date, test_ratio

const = Constants()
test_times = 1

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
    const.STOCK_PRICE: {const.DATA_PERIOD: 1},
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
        const.ONE_MONTH,
        const.ONE_WEEK,
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
stock_list = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK', '0006.HK', '0011.HK', '0012.HK', '0016.HK',
              '0017.HK', '0019.HK', '0023.HK', '0027.HK', '0066.HK', '0083.HK', '0101.HK', '0135.HK', '0144.HK',
              '0151.HK', '0267.HK', '0293.HK', '0386.HK', '0388.HK', '0494.HK', '0688.HK', '0700.HK', '0762.HK',
              '0823.HK', '0836.HK', '0857.HK', '0883.HK', '0939.HK', '0941.HK', '0992.HK', '1038.HK', '1044.HK',
              '1088.HK', '1109.HK', '1299.HK', '1398.HK', '1880.HK', '1928.HK', '2018.HK', '2318.HK', '2319.HK',
              '2388.HK', '2628.HK', '3328.HK', '3988.HK', '6823.HK']

# amount_method_list = [const.ARTIFICIAL_NEURAL_NETWORK, const.LINEAR_REGRESSION, const.RANDOM_FOREST]
amount_method_list = [const.LINEAR_REGRESSION, const.LINEAR_REGRESSION, const.LINEAR_REGRESSION]
trend_method_list = [const.SVM, const.RANDOM_FOREST, const.LOGISTIC_REGRESSION]

test = None
for amount_method, trend_method in zip(amount_method_list, trend_method_list):
    # for trend_method in trend_method_list:
    method = '{}_{}'.format(amount_method.split(('_'))[0].lower(), trend_method.split('_')[0].lower())
    new_file_path = os.path.join(output_path, method)
    if not os.path.isdir(new_file_path):
        os.makedirs(new_file_path)

    f = open(os.path.join(new_file_path, "stock_info.csv"), 'w')
    f.write('stock,MSE,MAPE,MAD,RMSE,CDC,HMSE,ME,test_time\n')
    # for stock in stock_list[:10]:
    start_time = time.time()
    for stock in stock_list:
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
            test = MixInferenceSystem(stock, amount_type=const.RATIO_AMOUNT, data_folder_path=data_path,
                                      using_exist_model=False, amount_method=amount_method,
                                      direction_method=trend_method, output_file_path=specific_file_path,
                                      model_path=specific_model_path)
            predict_result = test.predict_historical_data(test_start_date=ratio, start_date=date_start,
                                                          end_date=date_end, iterations=10)
            try:
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
                time.sleep(60)

        f.write('{},{},{},{},{},{},{},{},{}\n'.format(stock, mse / test_times, mape / test_times, mad / test_times,
                                                      rmse / test_times, cdc / test_times, hmse / test_times,
                                                      me / test_times, time.time() - start_time))

    f.close()

if hasattr(test, 'sc'):
    test.sc.stop()
