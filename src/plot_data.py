#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: plot_data
# Author: Mark Wang
# Date: 25/4/2016

import numpy as np


symbol_list = ['ro-', 'b^-', 'y*-', 'ms-']


def plot_predict_and_real(data, graph_index=0, graph_title=None, plt=None):
    # if plt is None:
    #     import matplotlib.pyplot as plt
    # plt.figure(graph_index)
    # plt.title(graph_title)
    # plt.xlabel("Stock Price")
    # plt.ylabel("Date")
    # predict_price = []
    # real_price = []
    # for i, j in data:
    #     real_price.append(i)
    #     predict_price.append(j)
    # predict = plt.plot(predict_price, 'ro-')[0]
    # real = plt.plot(real_price, 'g^-')[0]
    # plt.legend([predict, real], ["Predict Price", "Real Price"], loc=2)
    return plot_label_vs_data(data, ["Predict Price", "Real Price"], graph_index=graph_index,
                              graph_title=graph_title, plt=plt)


def plot_label_vs_data(data, label, graph_index=0, graph_title=None, plt=None):
    if plt is None:
        import matplotlib.pyplot as plt
    plt.figure(graph_index)
    plt.title(graph_title)
    plt.ylabel("Stock Price")
    plt.xlabel("Date")
    data = np.array(data)
    data = data.T
    plt_list = []
    for i in range(len(data)):
        plt_list.append(plt.plot(data[i], symbol_list[i])[0])
    plt.legend(plt_list, label, loc=2)
    return plt
