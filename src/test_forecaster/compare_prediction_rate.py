#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: compare_prediction_rate
# Author: Mark Wang
# Date: 26/10/2016

import os

import pandas as pd
import numpy as np

root_path = '/Users/warn/Documents/MSc(CS) Dissertation/Some Data/DataFrame'
data_path = os.path.join(root_path, 'ImproveDirectionSuccessfulRate')


def get_performance(path):
    result_df = pd.DataFrame(columns=['sdpr', 'mse', 'mape', 'time'])
    for dir_info in os.listdir(path):
        if not (dir_info.endswith('Fasle') or dir_info.endswith('True')):
            continue

        df = pd.read_csv(os.path.join(path, dir_info, 'statistics.csv'),
                         index_col=0)
        result_df.loc[dir_info] = df.mean()

    return result_df
