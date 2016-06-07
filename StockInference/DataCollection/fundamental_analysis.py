#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: fundamental_analysis
# Author: Mark Wang
# Date: 1/6/2016

from urllib import urlencode
from urllib2 import Request, urlopen

import numpy as np

from StockInference.DataCollection.base_class import BaseClass


class FundamentalAnalysis(BaseClass):
    def __init__(self):
        BaseClass.__init__(self)
        self._bond_label_dict = {
            self.US10Y_BOND: "^TNX",
            self.US30Y_BOND: "^TYX",
            self.HSI: "^HSI",
            self.FXI: "FXI",
            self.IC: "2801.HK",
            self.IA: "2829.HK",
            self.IA300: "2846.HK",
            self.IMSCI: "3010.HK",
        }
        self.fa_pca_transformer = None
        self.fa_min_list = []
        self.fa_max_list = []

    def _get_bond_price(self, symbol):
        start_date = self._start_date.split('-')
        end_date = self._true_end_date.split('-')
        data_list = [('s', symbol),
                     ('a', int(start_date[1]) - 1),
                     ('b', start_date[2]),
                     ('c', start_date[0]),
                     ('d', int(end_date[1]) - 1),
                     ('e', end_date[2]),
                     ('f', end_date[0]),
                     ('g', 'd')
                     ]
        url = "http://ichart.finance.yahoo.com/table.csv?{}".format(urlencode(data_list))
        query = Request(url)
        response = urlopen(query)
        bond_info = response.read()
        bond_info = [i.split(',') for i in bond_info.split('\n')][1:-1]
        bond_price = {}
        for i in bond_info:
            bond_price[i[0]] = float(i[4])
        return bond_price

    def fa_pca_data_reduction(self, data):
        if self.fa_pca_transformer is None:
            self.fa_pca_transformer = self.get_pca_transformer(data, 'mle')
        Y = self.fa_pca_transformer.transform(data)
        return Y

    def fa_data_normalization(self, data):
        pca_data = self.fa_pca_data_reduction(data)
        if not self.fa_min_list or not self.fa_max_list:
            n_y = len(pca_data[0])
            self.fa_max_list = np.zeros(n_y)
            self.fa_min_list = np.zeros(n_y)
            for i in range(n_y):
                self.fa_min_list[i] = np.min(pca_data[:, i])
                self.fa_max_list[i] = np.max(pca_data[:, i])
        diff = self.fa_max_list - self.fa_min_list
        nor_data = map(lambda p: ((p - self.fa_min_list) / diff).tolist(), pca_data)

        return nor_data

    def fundamental_analysis(self, required_info):
        return self.fa_data_normalization(self.raw_fundamental_analysis(required_info))

    def raw_fundamental_analysis(self, required_info):
        if not self._date_list:
            self.generate_date_list()
        calculated_info = [[i] for i in self._date_list]
        for info in required_info:
            if info in self._bond_label_dict:
                bond_price = self._get_bond_price(self._bond_label_dict[info])

                calculated_info = self._merge_info(calculated_info=calculated_info, info_dict=bond_price)

        return [i[1:] for i in calculated_info]

    def raw_fundamental_analysis_change_rate(self, required_info):
        if not self._date_list:
            self.generate_date_list()

        calculated_info = [[i] for i in self._date_list]
        for info in required_info:
            if info in self._bond_label_dict:
                bond_price = self._get_bond_change_rate(self._bond_label_dict[info])

                calculated_info = self._merge_info(calculated_info=calculated_info, info_dict=bond_price)

        return [i[1:] for i in calculated_info]

    def _get_bond_change_rate(self, symbol):
        start_date = self.get_ahead_date(self._start_date, 10).split('-')
        end_date = self._true_end_date.split('-')
        data_list = [('s', symbol),
                     ('a', int(start_date[1]) - 1),
                     ('b', start_date[2]),
                     ('c', start_date[0]),
                     ('d', int(end_date[1]) - 1),
                     ('e', end_date[2]),
                     ('f', end_date[0]),
                     ('g', 'd')
                     ]
        url = "http://ichart.finance.yahoo.com/table.csv?{}".format(urlencode(data_list))
        query = Request(url)
        response = urlopen(query)
        bond_info = response.read()
        bond_info = [i.split(',') for i in bond_info.split('\n')][1:-1]
        bond_price_index = {}
        bond_price_map = {}
        for i in range(len(bond_info)):
            bond_price_index[bond_info[i][0]] = i

        for date in self._date_list:
            if date not in bond_price_index:
                bond_price_map[date] = 0
            else:
                today_index = bond_price_index[date]
                today_price = float(bond_info[today_index][4])
                last_day_price = float(bond_info[today_index + 1][4])
                bond_price_map[date] = (today_price - last_day_price) / last_day_price

        return bond_price_map