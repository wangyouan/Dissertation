#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: fix_file_info
# Author: Mark Wang
# Date: 30/10/2016

import os

import pandas as pd


def fix_file_info():
    path = '/Users/warn/Documents/MSc(CS) Dissertation/Some Data/output'

    for x, y, z in os.walk(path):
        if 'stock_info.csv' not in z:
            continue

        with open(os.path.join(x, 'stock_info.csv')) as f:
            info = f.read()

        lines = info.split('\n')
        new_lines = [lines[0]]

        for line in lines[1:]:
            info = line.split(',')
            time_mse = info[-1].split('.')
            info[-1] = '.'.join(time_mse[:2])
            info.append('.'.join(time_mse[2:]))
            new_lines.append(','.join(info))

        with open(os.path.join(x, 'stock_info_new.csv'), 'w') as f:
            f.write('\n'.join(new_lines))


def calculate_date():
    path = '/Users/warn/Documents/MSc(CS) Dissertation/Some Data/output'

    result_df = None

    for x, y, z in os.walk(path):
        if 'stock_info_new.csv' not in z:
            continue

        df = pd.read_csv(os.path.join(x, 'stock_info_new.csv'), index_col=0)
        if result_df is None:
            result_df = pd.DataFrame(columns=df.keys())

        result_df.loc[x.split('/')[-1]] = df.mean()

    result_df.to_csv(os.path.join(path, 'statistic.csv'))

