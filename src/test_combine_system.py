#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: test_combine_system
# Author: Mark Wang
# Date: 18/6/2016


import os
import sys
import time

from StockInference.composition_prediction_system import MixInferenceSystem
from StockInference.util.data_parse import *
from StockInference.constant import Constants
from __init__ import start_date, end_date, test_ratio

const = Constants()
test_times = 3

if len(sys.argv) > 3:
    date_start = sys.argv[1]
    date_end = sys.argv[2]
    ratio = sys.argv[3]
else:
    date_start = start_date
    date_end = end_date
    ratio = test_ratio

required_info = {
    const.PRICE_TYPE: const.STOCK_CLOSE,
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
stock_list = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK', '0006.HK', '0007.HK', '0008.HK', '0009.HK',
              '0010.HK', '0011.HK', '0012.HK', '0013.HK', '0014.HK', '0015.HK', '0016.HK', '0017.HK', '0018.HK',
              '0019.HK', '0020.HK', '0021.HK', '0022.HK', '0023.HK', '0024.HK', '0025.HK', '0026.HK', '0027.HK',
              '0028.HK', '0029.HK', '0030.HK', '0031.HK', '0032.HK', '0700.HK', '0034.HK', '0035.HK', '0036.HK',
              '0068.HK', '0038.HK', '0039.HK', '0040.HK', '0041.HK', '0042.HK', '0043.HK', '0044.HK', '0045.HK',
              '0046.HK', '0088.HK', '0050.HK', '0051.HK', '0052.HK', '0053.HK', '0054.HK', '0168.HK', '0056.HK',
              '0057.HK', '0058.HK', '0059.HK', '0060.HK', '0888.HK', '0062.HK', '0063.HK', '0064.HK', '0065.HK',
              '0066.HK', '1123.HK']

amount_method_list = [const.RANDOM_FOREST, const.LINEAR_REGRESSION, const.ARTIFICIAL_NEURAL_NETWORK]
trend_method_list = [const.SVM, const.LOGISTIC_REGRESSION, const.RANDOM_FOREST]

test = None
for amount_method, trend_method in zip(amount_method_list, trend_method_list):
    # for trend_method in trend_method_list:
    method = '{}_{}'.format(amount_method.split(('_'))[0].lower(), trend_method.split('_')[0].lower())
    new_file_path = os.path.join(output_path, method)
    if not os.path.isdir(new_file_path):
        os.makedirs(new_file_path)

    f = open(os.path.join(new_file_path, "stock_info.csv"), 'w')
    f.write('stock,MSE,MAPE,MAD,RMSE,CDC,HMSE,ME\n')
    for stock in stock_list[:10]:
    # for stock in ['0377.HK', '1195.HK']:
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
            test = MixInferenceSystem(stock, amount_type=const.RAW_AMOUNT, data_folder_path=data_path,
                                      using_exist_model=False, amount_method=amount_method,
                                      direction_method=trend_method, output_file_path=specific_file_path,
                                      model_path=specific_model_path)
            try:
                predict_result = test.predict_historical_data(test_start_date=ratio, start_date=date_start,
                                                              end_date=date_end, iterations=10)
                predict_result_rdd = test.sc.parallelize(predict_result)
                me += get_ME(predict_result_rdd)
                mse += get_MSE(predict_result_rdd)
                mape += get_MAPE(predict_result_rdd)
                mad += get_MAD(predict_result_rdd)
                rmse += get_RMSE(predict_result_rdd)
                hmse += get_HMSE(predict_result_rdd)
                # tie = get_theils_inequality_coefficient(predict_result)
                cdc += get_CDC_combine(predict_result_rdd)
            except Exception, err:
                print "Error happens"
                print err
                time.sleep(60)

        f.write('{},{},{},{},{},{},{},{}\n'.format(stock, mse / test_times, mape / test_times, mad / test_times,
                                                   rmse / test_times, cdc / test_times, hmse / test_times,
                                                   me / test_times))

    f.close()

if hasattr(test, 'sc'):
    test.sc.stop()
