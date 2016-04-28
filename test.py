#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: test
# Author: Mark Wang
# Date: 25/4/2016

import os
import logging
import sys

import matplotlib.pyplot as plt

from src import load_spark_context
from src.linear_regression_with_SGD import calculate_data_normalized
from src.RandomForestRegression import price_predict
from src.plot_data import plot_label_vs_data

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


def compare_lr_and_rt(stock_num=5, window_size=5, show_price_prediction=False):
    column = ["Real", "LinearRegression", "RandomForest"]
    sc = load_spark_context()[0]
    index = 0
    folder, path, files = os.walk('data').next()
    for file in files:
        if not stock_num:
            logger.info("Enough stock number, stop calculating")
            break

        if 'HK' not in file and not file.endswith('csv') and not file.startswith('00'):
            logger.debug("File name {}, not suitable for this model, continue".format(file))
            continue

        try:
            logger.debug("Load data from {}".format(file))
            data_file_path = os.path.join(os.path.abspath(os.curdir), 'data/{}'.format(file))
            linear_sgd = calculate_data_normalized(path=data_file_path, windows=window_size, spark_context=sc)
            ran_tree = price_predict(data_file_path, windows=window_size, spark_context=sc)
            close_price = linear_sgd[0].zip(ran_tree[0]).map(lambda (x, y): (x[0], x[1], y[1])).take(100)
            open_price = linear_sgd[1].zip(ran_tree[1]).map(lambda (x, y): (x[0], x[1], y[1])).take(100)
            if show_price_prediction:
                logger.debug("Plot {} of {} days".format(file, window_size))
                plot_label_vs_data(data=close_price, label=column, graph_index=index,
                                   graph_title="{} close price compare (using last {} days)"
                                   .format('.'.join(file.split('.')[:-1]), window_size), plt=plt)
                plot_label_vs_data(data=open_price, label=column, graph_index=index + 1,
                                   graph_title="{} open price compare (using last {} days)"
                                   .format('.'.join(file.split('.')[:-1]), window_size), plt=plt)
                index += 2
        except Exception, e:
            logging.warning("Unable to calculate file {}, as {}".format(file, e))
        else:
            stock_num-=1
    if show_price_prediction:
        plt.show()


if __name__ == '__main__':
    compare_lr_and_rt()