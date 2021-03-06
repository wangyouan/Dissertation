#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: test_ann_hidden_nodes_num
# Author: Mark Wang
# Date: 2/11/2016

import os
import time

import pandas as pd

from stockforecaster import predict_stock_price_spark
from stockforecaster import StockForecaster as SF
from stockforecaster.util.evaluate_func import calculate_mean_squared_error, \
    calculate_success_direction_prediction_rate, calculate_mean_absolute_percentage_error

start_date = '2013-07-06'
end_date = '2016-01-06'
test_date = '2015-01-06'
worker_number = 2
window_size = 6
if os.uname()[1] == 'ewin3011':
    root_path = '/home/wangzg/Documents/WangYouan/.dissertation/Dissertation'
elif os.uname()[1] == 'Master':
    root_path = '/home/hadoop/Projects/Dissertation'
elif os.uname()[1].startswith('warn-Inspiron'):
    root_path = '/home/warn/PythonProjects/Dissertation'
else:
    root_path = '/Users/warn/PycharmProjects/Dissertation'

data_path = os.path.join(root_path, 'data')
result_path = os.path.join(root_path, 'result')

hsi_stock_list = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK', '0006.HK', '0011.HK', '0012.HK', '0016.HK',
                  '0017.HK', '0019.HK', '0023.HK', '0027.HK', '0066.HK', '0083.HK', '0101.HK', '0135.HK', '0144.HK',
                  '0151.HK', '0267.HK', '0293.HK', '0386.HK', '0388.HK', '0494.HK', '0688.HK', '0700.HK', '0762.HK',
                  '0823.HK', '0836.HK', '0857.HK', '0883.HK', '0939.HK', '0941.HK', '0992.HK', '1038.HK', '1044.HK',
                  '1088.HK', '1109.HK', '1299.HK', '1398.HK', '1880.HK', '1928.HK', '2018.HK', '2318.HK', '2319.HK',
                  '2388.HK', '2628.HK', '3328.HK', '3988.HK', '6823.HK']

short_name_dict = {SF.ARTIFICIAL_NEURAL_NETWORK: 'ann',
                   SF.LINEAR_REGRESSION: 'lrc',
                   SF.LOGISTIC_REGRESSION: 'lrr',
                   SF.RANDOM_FOREST: 'rt'}

if __name__ == '__main__':

    for hidden_nodes_nums in [30]:

        path = os.path.join(result_path, 'hidden', '{}'.format(hidden_nodes_nums))

        for train_method in [
            # {SF.CHANGE_AMOUNT: SF.ARTIFICIAL_NEURAL_NETWORK,
            #  SF.CHANGE_DIRECTION: SF.ARTIFICIAL_NEURAL_NETWORK},
            # {SF.CHANGE_AMOUNT: SF.LINEAR_REGRESSION,
            #  SF.CHANGE_DIRECTION: SF.RANDOM_FOREST},
            # {SF.CHANGE_AMOUNT: SF.RANDOM_FOREST,
            #  SF.CHANGE_DIRECTION: SF.LOGISTIC_REGRESSIONr},
            SF.ARTIFICIAL_NEURAL_NETWORK,
            # SF.LINEAR_REGRESSION, SF.RANDOM_FOREST,
        ]:
            df = pd.DataFrame(columns=['stock', 'sdpr', 'mse', 'mape', 'time'])
            df1 = pd.DataFrame(columns=['stock', 'sdpr', 'mse', 'mape', 'time'])

            # print train_method
            if isinstance(train_method, dict):
                current_result_path = os.path.join(path,
                                                   "{}_{}".format(short_name_dict[train_method[SF.CHANGE_DIRECTION]],
                                                                  short_name_dict[train_method[SF.CHANGE_AMOUNT]]))

                current_result_path1 = os.path.join(path, "{}_{}_True"
                                                    .format(short_name_dict[train_method[SF.CHANGE_DIRECTION]],
                                                            short_name_dict[train_method[SF.CHANGE_AMOUNT]]))
            else:
                current_result_path = os.path.join(path, short_name_dict[train_method])

                current_result_path1 = current_result_path

            print current_result_path
            if not os.path.isdir(current_result_path):
                os.makedirs(current_result_path)

            if not os.path.isdir(current_result_path1):
                os.makedirs(current_result_path1)

            for i in range(len(hsi_stock_list)):
                start_time = time.time()
                stock = hsi_stock_list[i]
                print 'start to get stock', stock
                save_file_name = '{}.csv'.format(stock[:4])

                try:

                    result = predict_stock_price_spark(stock_symbol=stock, data_path=data_path,
                                                       worker_num=worker_number, hidden_nodes_num=hidden_nodes_nums,
                                                       train_method=train_method, start_date=start_date,
                                                       end_date=end_date, using_percentage=False,
                                                       test_date=test_date, window_size=window_size)
                except Exception, err:
                    import traceback

                    traceback.print_exc()
                    print stock
                    break

                else:
                    result[['Target', 'TodayPrice', 'prediction']].to_csv(
                        os.path.join(current_result_path, save_file_name))

                    df.loc[i] = {
                        'sdpr': calculate_success_direction_prediction_rate(result, SF.TODAY_PRICE, 'prediction',
                                                                            SF.TARGET_PRICE),
                        'mse': calculate_mean_squared_error(result, 'prediction', SF.TARGET_PRICE),
                        'mape': calculate_mean_absolute_percentage_error(result, 'prediction', SF.TARGET_PRICE),
                        'stock': stock,
                        'time': time.time() - start_time}

                if not isinstance(train_method, dict):
                    continue

                start_time = time.time()
                try:

                    result = predict_stock_price_spark(stock_symbol=stock, data_path=data_path,
                                                       worker_num=worker_number, hidden_nodes_num=hidden_nodes_nums,
                                                       train_method=train_method, start_date=start_date,
                                                       end_date=end_date, using_percentage=True,
                                                       test_date=test_date, window_size=window_size)
                except Exception, err:
                    import traceback

                    traceback.print_exc()
                    print stock
                    break

                else:
                    result[['Target', 'TodayPrice', 'prediction']].to_csv(
                        os.path.join(current_result_path1, save_file_name))

                    df1.loc[i] = {
                        'sdpr': calculate_success_direction_prediction_rate(result, SF.TODAY_PRICE, 'prediction',
                                                                            SF.TARGET_PRICE),
                        'mse': calculate_mean_squared_error(result, 'prediction', SF.TARGET_PRICE),
                        'mape': calculate_mean_absolute_percentage_error(result, 'prediction', SF.TARGET_PRICE),
                        'stock': stock,
                        'time': time.time() - start_time}

            df.to_csv(os.path.join(current_result_path, 'statistics.csv'), index=False)
            if not df1.empty:
                df1.to_csv(os.path.join(current_result_path1, 'statistics.csv'), index=False)
