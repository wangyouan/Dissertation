#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: plot_data
# Author: Mark Wang
# Date: 25/4/2016

import numpy as np


def plot_predict_and_real(data, graph_index=0, graph_title=None, plt=None):
    if plt is None:
        import matplotlib.pyplot as plt
    plt.figure(graph_index)
    plt.title(graph_title)
    plt.xlabel("Stock Price")
    plt.ylabel("Date")
    predict_price = []
    real_price = []
    for i, j in data:
        real_price.append(i)
        predict_price.append(j)
    predict, = plt.plot(predict_price, 'ro-')
    real, = plt.plot(real_price, 'g^-')
    plt.legend([predict, real], ["Predict Price", "Real Price"], loc=2)
    return plt