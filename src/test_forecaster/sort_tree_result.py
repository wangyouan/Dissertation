#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: sort_tree_result
# Author: Mark Wang
# Date: 3/11/2016


import os

import pandas as pd


def sort_tree_result():
    root_path = '/Users/warn/Documents/MSc(CS) Dissertation/Some Data/DataFrame/trees_num'

    rt_df = pd.DataFrame(columns=['mse', 'mape', 'cdc', 'time'])
    rt_rt_df = pd.DataFrame(columns=['mse', 'mape', 'cdc', 'time'])
    rt_rt_true_df = pd.DataFrame(columns=['mse', 'mape', 'cdc', 'time'])

    for directory in os.listdir(root_path):
        current_path = os.path.join(root_path, directory)
        if directory.startswith('.') or not os.path.isdir(current_path):
            continue

        for x, y, z in os.walk(current_path):
            if 'statistics.csv' not in z:
                continue

            tmp_df = pd.read_csv(os.path.join(x, 'statistics.csv'), index_col=0)
            tmp_df = tmp_df[tmp_df.index != '^HSI']
            tmp_df['cdc'] = tmp_df['sdpr']
            del tmp_df['sdpr']
            if x.endswith('True') or x.endswith('true'):
                rt_rt_true_df.loc[directory] = tmp_df.mean()

            elif x.endswith('rt_rt'):
                rt_rt_df.loc[directory] = tmp_df.mean()

            else:
                rt_df.loc[directory] = tmp_df.mean()

    return rt_df, rt_rt_df, rt_rt_true_df


def sort_ann_result():
    root_path = '/Users/warn/Documents/MSc(CS) Dissertation/Some Data/DataFrame/hidden'

