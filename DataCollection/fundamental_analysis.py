#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: fundamental_analysis
# Author: Mark Wang
# Date: 1/6/2016

from urllib import urlencode
from urllib2 import Request, urlopen

from base_class import BaseClass


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
        bond_info = [(i[0], i[4]) for i in bond_info]
        return bond_info

    def fundamental_analysis(self, required_info):
        pass