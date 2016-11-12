#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: draw_picture
# Author: Mark Wang
# Date: 11/11/2016

import os

import matplotlib.pyplot as plt

import pandas as pd


def draw_picture(path, save_path, title):
    df = pd.read_csv(path, index_col=0)

    label_dict = {'sdpr': 'CDC',
                  'mse': 'MSE',
                  'mape': 'MAPE'}

    for key in ['sdpr', 'mse', 'mape']:
        current_title = '{} {} result'.format(title, label_dict[key])
        file_name = '{}{}.png'.format(''.join([i[0] for i in title.split(' ')]), label_dict[key])
        fig, ax1 = plt.subplots()
        ax1.plot(range(len(df.index)), df[key], 'b-', label=label_dict[key])
        ax1.set_ylabel(label_dict[key])
        ax1.set_xlabel('Stock Symbol')

        ax2 = ax1.twinx()
        ax2.bar(range(len(df.index)), df['avgPrice'], width=0.5, color='r', label='Average Stock Price')
        ax2.set_ylabel('Average Stock Price')

        ax1.set_title(current_title)

        xticks = list(df.index)

        for i in range(len(xticks)):
            if i % 2 == 1:
                xticks[i] = ''

        plt.gca().xaxis.set_major_locator(plt.NullFormatter())
        plt.xticks(range(len(df.index)), xticks, rotation='vertical', size=5)
        # plt.margins(0.2)
        # plt.subplots_adjust(bottom=0.15)
        plt.legend(loc=0)
        for ax in fig.axes:
            plt.sca(ax)
            plt.xticks(rotation=60, size='small')
        fig.savefig(os.path.join(save_path, file_name))


if __name__ == '__main__':

    save_path = '/Users/warn/Documents/Projects/DissertationReport/graduate-thesis/Figures/TestingResult'

    file_name_dict = {'ann': 'Artificial Neural Network',
                      'ann_ann': 'Artificial Neural Network + Artificial Neural Network',
                      'rt': 'Random Forest',
                      'rt_lrc': 'Random Forest + Linear Regression',
                      'lrr_rt': 'Logistic Regression + Random Forest',
                      'lrc': 'Linear Regression'}
    for x, y, z in os.walk('/Users/warn/Documents/MScDissertation/Data/DataFrame/start_date/2013-07-06'):
        if 'statisticsAddPrice.csv' not in z:
            continue

        algorithm = x.split('/')[-1]
        draw_picture(os.path.join(x, 'statisticsAddPrice.csv'), save_path, file_name_dict[algorithm])
