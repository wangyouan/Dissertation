#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: test
# Author: Mark Wang
# Date: 25/4/2016

import os

import pandas as pd
import matplotlib.pyplot as plt

from src import load_spark_context
from src.linear_regression_with_SGD import calculate_data_normalized
from src.RandomForestRegression import price_predict
from src.plot_data import plot_label_vs_data


def compare_lr_and_rt():
    sc = load_spark_context()[0]
    data_file_path = os.path.join(os.path.abspath(os.curdir), 'data/0001.HK.csv')
    window_size = 5
    linear_sgd = calculate_data_normalized(path=data_file_path, windows=window_size, spark_context=sc)
    ran_tree = price_predict(data_file_path, windows=window_size, spark_context=sc)
    close_price = linear_sgd[0].zip(ran_tree[0]).map(lambda (x, y): (x[0], x[1], y[1])).take(100)
    open_price = linear_sgd[1].zip(ran_tree[1]).map(lambda (x, y): (x[0], x[1], y[1])).take(100)
    print close_price, open_price
    open_df = pd.DataFrame(open_price, columns=["real", "linear", "random"])
    close_df = pd.DataFrame(close_price, columns=["real", "linear", "random"])
    plot_label_vs_data(data=close_price, label=["real", "linear", "random"], graph_index=0,
                       graph_title="close price compare", plt=plt)
    plot_label_vs_data(data=open_price, label=["real", "linear", "random"], graph_index=1,
                       graph_title="open price compare", plt=plt)
    plt.show()


if __name__ == '__main__':
    compare_lr_and_rt()