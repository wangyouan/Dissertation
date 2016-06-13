#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: stock_prediction
# Author: Mark Wang
# Date: 13/6/2016

import os
import sys
import time

from StockInference.inference_system import InferenceSystem
from StockInference.util.data_parse import get_MAD, get_MAPE, get_MSE

output_path = 'output'
data_path = 'data'

if sys.platform == 'darwin':
    output_path = '../{}'.format(output_path)
    data_path = '../{}'.format(data_path)

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

for method in [InferenceSystem.ARTIFICIAL_NEURAL_NETWORK, InferenceSystem.RANDOM_FOREST,
               InferenceSystem.LINEAR_REGRESSION][:1]:

    new_file_path = os.path.join(output_path, method)
    if not os.path.isdir(new_file_path):
        os.makedirs(new_file_path)

    f = open(os.path.join(new_file_path, "stock_info.csv"), 'w')
    f.write('stock,MSE,MAPE,MAD\n')
    for stock in stock_list:
        # for stock in ["0033.HK"]:
        specific_file_path = os.path.join(new_file_path, stock[:4])
        test = InferenceSystem(stock)
        predict_result = test.predict_historical_data(0.8, "2006-04-14", "2016-04-15",
                                                      training_method=method,
                                                      data_folder_path=data_path,
                                                      output_file_path=specific_file_path,
                                                      load_model=False)
        mse = get_MSE(predict_result)
        mape = get_MAPE(predict_result)
        mad = get_MAD(predict_result)
        f.write('{},{},{},{}\n'.format(stock, mse, mape, mad))
        test.sc.stop()
        time.sleep(30)

    f.close()
