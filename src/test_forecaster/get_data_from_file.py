#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: get_data_from_file
# Author: Mark Wang
# Date: 3/11/2016

import os

import pandas as pd
from sklearn.metrics import mean_squared_error


def get_mse(path_or_df):
    if isinstance(path_or_df, str):
        df = pd.read_csv(path_or_df, index_col=0)

    else:
        df = path_or_df

    return mean_squared_error(df['Target'], df['prediction'])


def get_cdc(path_or_df):
    if isinstance(path_or_df, str):
        df = pd.read_csv(path_or_df, index_col=0)

    else:
        df = path_or_df

    today = df['TodayPrice']
    target = df['Target']
    prediction = df['prediction']
    return float((((target - today) * (prediction - today)) > 0).sum()) / today.values.shape[0]


def get_mape(path_or_df):
    if isinstance(path_or_df, str):
        df = pd.read_csv(path_or_df, index_col=0)

    else:
        df = path_or_df

    target = df['Target']
    prediction = df['prediction']
    return ((prediction - target).apply(abs) / prediction).mean()


if __name__ == '__main__':
    root_path = '/Users/warn/Documents/MSc(CS) Dissertation/Some Data/DataFrame/trees_num'

    for x, y, z in os.walk(root_path):
        if 'rt_rt_True' not in y:
            continue

        data_df = pd.read_csv(os.path.join(x, 'rt_rt_True', 'statistics.csv'), index_col=0)

        for i in data_df.index:
            data_df.ix[i, 'sdpr'] = get_cdc(os.path.join(x, 'rt_rt_True', '{}.csv'.format(i[:4])))
            data_df.ix[i, 'mse'] = get_mse(os.path.join(x, 'rt_rt_True', '{}.csv'.format(i[:4])))
            data_df.ix[i, 'mape'] = get_mape(os.path.join(x, 'rt_rt_True', '{}.csv'.format(i[:4])))

        data_df.to_csv(os.path.join(x, 'rt_rt_True', 'statistics.csv'))
