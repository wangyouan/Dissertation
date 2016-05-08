#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: test
# Author: Mark Wang
# Date: 25/4/2016

import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

from src import load_spark_context
from src.RandomForestRegression import price_predict
from src.linear_regression_with_SGD import calculate_data_normalized
from src.parse_data import DataParser
from src.plot_data import plot_label_vs_data, plot_bar

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stock_calculate = {}
if os.path.exists('stock_data'):
    f = open('stock_data')
    stock_calculate = pickle.load(f)
    f.close()


sc = load_spark_context()[0]


def compare_lr_and_rt(stock_num=None, window_size=5, show_price_prediction=False):
    column = ["Real", "LinearRegression", "RandomForest"]
    index = 0
    folder, path, files = os.walk('data').next()
    if stock_num is None:
        stock_num = len(files)
    for file in files:
        if not stock_num:
            logger.info("Enough stock number, stop calculating")
            break

        if 'HK' not in file or not file.endswith('csv') or not file.startswith('00'):
            logger.debug("File name {}, not suitable for this model, continue".format(file))
            continue

        stock_symbol = '.'.join(file.split('.')[:-1])

        logger.info("Start to calculate {} of {} days".format(stock_symbol, window_size))
        if stock_symbol not in stock_calculate:
            stock_calculate[stock_symbol] = {}

        try:
            logger.debug("Load data from {}".format(file))
            data_file_path = os.path.join(os.path.abspath(os.curdir), 'data/{}'.format(file))
            linear_sgd = calculate_data_normalized(path=data_file_path, windows=window_size, spark_context=sc)
            ran_tree = price_predict(data_file_path, windows=window_size, spark_context=sc)
            close_price = linear_sgd[0].zip(ran_tree[0]).map(lambda (x, y): (x[0], x[1], y[1])).take(100)
            open_price = linear_sgd[1].zip(ran_tree[1]).map(lambda (x, y): (x[0], x[1], y[1])).take(100)
            stock_calculate[stock_symbol][window_size] = {
                "lr_MSE": DataParser.get_MSE(linear_sgd[1]),
                "lr_MAPE": DataParser.get_MAPE(linear_sgd[1]),
                "lr_MAD": DataParser.get_MAD(linear_sgd[1]),
                "rf_MAD": DataParser.get_MAD(ran_tree[1]),
                "rf_MAPE": DataParser.get_MAPE(ran_tree[1]),
                "rf_MSE": DataParser.get_MSE(ran_tree[1]),
            }
            # stock_calculate[stock_symbol][window_size]["rf_MAD"] = DataParser.get_MAD(ran_tree[1])
            # stock_calculate[stock_symbol][window_size]["rf_MAPE"] = DataParser.get_MAPE(ran_tree[1])
            # stock_calculate[stock_symbol][window_size]["rf_MSE"] = DataParser.get_MSE(ran_tree[1])

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
            stock_num -= 1
    if show_price_prediction:
        plt.show()


def handle_stock_data():
    n_days = [None] * 10
    stock_number = 0
    unused_key = []
    for key in stock_calculate:
        for day in stock_calculate[key]:
            for i in stock_calculate[key][day]:
                if np.isnan(stock_calculate[key][day][i]):
                    unused_key.append(key)
                    break

    for key in unused_key:
        if key in stock_calculate:
            del stock_calculate[key]

    # import pprint
    # pprint.pprint(stock_calculate)

    for key in stock_calculate:
        if stock_number == 60:
            break
        f1 = open('output/{}_MAPE.csv'.format(key), 'w')
        f2 = open('output/{}_MSE.csv'.format(key), 'w')
        f3 = open('output/{}_MAD.csv'.format(key), 'w')
        f1.write('N days,LinearRegression,RandomForest\n')
        f2.write('N days,LinearRegression,RandomForest\n')
        f3.write('N days,LinearRegression,RandomForest\n')
        stock_number += 1
        for day in stock_calculate[key]:
            f1.write(
                '{},{},{}\n'.format(day, stock_calculate[key][day]['lr_MAPE'], stock_calculate[key][day]['rf_MAPE']))
            f2.write(
                '{},{},{}\n'.format(day, stock_calculate[key][day]['lr_MSE'], stock_calculate[key][day]['rf_MSE']))
            f3.write(
                '{},{},{}\n'.format(day, stock_calculate[key][day]['lr_MAD'], stock_calculate[key][day]['rf_MAD']))
            features = stock_calculate[key][day]
            if n_days[day - 1] is None:
                n_days[day - 1] = features.copy()
            else:
                for feature in features:
                    n_days[day - 1][feature] += features[feature]
        f1.close()
        f2.close()
        f3.close()

    f = open('output/average_MAPE.csv', 'w')
    f.write('N days,LinearRegression,RandomForest\n')
    lr_mape = []
    rt_mape = []
    for i in range(10):
        lr_mape.append(n_days[i]['lr_MAPE'] / stock_number)
        rt_mape.append(n_days[i]['rf_MAPE'] / stock_number)
        f.write('{},{},{}\n'.format(i + 1, n_days[i]['lr_MAPE'] / stock_number, n_days[i]['rf_MAPE'] / stock_number))

    f.close()
    plot_bar((lr_mape, rt_mape), ['LinearRegression', "RandomForest"], x_axis=(str(i) for i in range(1, 11)),
             y_label="Average MAPE",
             x_label="Days Number", plt=plt, graph_title="{} number stocks average MAPE compare".format(stock_number))
    plt.show()
    print n_days


if __name__ == '__main__':
    # for days in range(1, 11):
    #     compare_lr_and_rt(window_size=days)
    #
    # f = open("stock_data", 'w')
    # pickle.dump(stock_calculate, f)
    # f.close()
    # handle_stock_data()
    lr_mad = []
    rt_mad = []
    lr_mse = []
    rt_mse = []

    for day in range(1, 11):
        lr_mad.append(stock_calculate['0003.HK'][day]['lr_MAD'])
        rt_mad.append(stock_calculate['0003.HK'][day]['rf_MAD'])
        lr_mse.append(stock_calculate['0003.HK'][day]['lr_MSE'])
        rt_mse.append(stock_calculate['0003.HK'][day]['rf_MSE'])

    plot_bar((lr_mad, rt_mad), ['LinearRegression', "RandomForest"], x_axis=(str(i) for i in range(1, 11)),
             y_label="MAD", x_label="Days Number", plt=plt,
             graph_title="prediction of 0003.HK MAD compare")
    plot_bar((lr_mse, rt_mse), ['LinearRegression', "RandomForest"], x_axis=(str(i) for i in range(1, 11)),
             y_label="MSE", x_label="Days Number", plt=plt,
             graph_title="prediction of 0003.HK MSE compare")
    plt.show()
