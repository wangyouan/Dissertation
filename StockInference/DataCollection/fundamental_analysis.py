#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: fundamental_analysis
# Author: Mark Wang
# Date: 1/6/2016

import os
import pickle

import numpy as np
import quandl

from StockInference.DataCollection.base_class import BaseClass
from StockInference.util.get_history_stock_price import get_all_data_about_stock
from StockInference.util.date_parser import get_ahead_date
from StockInference.util.get_hk_interest_rate import get_hk_interest_rate

data_file_name = "interest_rate.dat"


def load_interest_rate():
    path = os.path.join(os.path.dirname(__file__), data_file_name)
    print path
    if os.path.isfile(path):
        f = open(path)
        raw_data = pickle.load(f)
        f.close()
        return raw_data
    else:
        raise Exception("cannot find data file location")
        return {}


def save_interest_rate(raw_data):
    try:
        f = open(os.path.join(os.path.dirname(__file__), data_file_name), 'w')
        pickle.dump(raw_data, f)
        f.close()
    except Exception, e:
        print "can not save data as %s" % e


class FundamentalAnalysis(BaseClass):
    def __init__(self):
        BaseClass.__init__(self)
        self._bond_label_dict_hk = {
            self.IC: "2801.HK",
            self.IA: "2829.HK",
            self.IA300: "2846.HK",
            self.IMSCI: "3010.HK",
        }
        self._bond_label_dict_us = {
            self.US10Y_BOND: "^TNX",
            self.US30Y_BOND: "^TYX",
            self.HSI: "^HSI",
            self.FXI: "FXI",
        }
        self.fa_pca_transformer = None
        self.fa_min_list = []
        self.fa_max_list = []
        quandl.ApiConfig.api_key = "RYdPmBZoFyLXxg1RQ3fY"

    def _get_bond_price(self, symbol, location=None):
        bond_info = get_all_data_about_stock(symbol=symbol, start_date=get_ahead_date(self.get_start_date(), 5),
                                             end_date=self.get_true_end_date(), remove_zero_volume=False)
        bond_price_map = {}
        for i in bond_info:
            bond_price_map[i[0]] = float(i[4])
        return bond_price_map

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

    def fundamental_analysis(self, required_info, fa_type=None):
        if fa_type is None or fa_type == self.FA_NORMALIZATION:
            return self.fa_data_normalization(self.raw_fundamental_analysis(required_info, ratio=False))
        elif fa_type == self.FA_RATIO:
            return self.raw_fundamental_analysis(required_info=required_info, ratio=True)
        elif fa_type == self.FA_RAW_DATA:
            return self.raw_fundamental_analysis(required_info=required_info, ratio=False)

    def raw_fundamental_analysis(self, required_info, ratio=False):
        if not self._date_list:
            self.generate_date_list()
        calculated_info = [[i] for i in self._date_list]
        for info in required_info:
            if isinstance(info, str):
                if info in self._bond_label_dict_hk:
                    bond_price = self._get_bond_price(self._bond_label_dict_hk[info])

                    calculated_info = self._merge_info(calculated_info=calculated_info, info_dict=bond_price)
                elif info in self._bond_label_dict_us:
                    bond_price = self._get_bond_price(self._bond_label_dict_us[info], location=self.UNITED_STATES)
                    calculated_info = self._merge_info(calculated_info=calculated_info, info_dict=bond_price)
                elif info in [self.ONE_MONTH, self.OVER_NIGHT, self.ONE_WEEK, self.ONE_YEAR, self.HALF_YEAR,
                              self.THREE_MONTHS, self.TWO_MONTHS]:
                    calculated_info = self._merge_info(calculated_info=calculated_info,
                                                       info_dict=self.get_interest_rate(info))
            elif isinstance(info, dict):
                if self.FROM in info:
                    currency_exchange_rate = self._get_currency_exchange_rate(info[self.FROM], info[self.TO])
                    calculated_info = self._merge_info(calculated_info=calculated_info,
                                                       info_dict=currency_exchange_rate)
                elif self.GOLDEN_PRICE in info:
                    golden_price = self._get_golden_price_in_cny(ratio=info.get(self.GOLDEN_PRICE, False))
                    calculated_info = self._merge_info(calculated_info=calculated_info, info_dict=golden_price)

        return [i[1:] for i in calculated_info]

    def _get_golden_price_in_cny(self, ratio=False):
        if not ratio:
            return self.get_quandl_data("WGC/GOLD_DAILY_CNY")
        else:
            data = self.get_quandl_data("WGC/GOLD_DAILY_CNY", transform='rdiff')
            return data

    def _get_currency_exchange_rate(self, from_cur, to_cur):
        rate_info = None
        if from_cur == self.EUR and to_cur == self.HKD:
            rate_info = self.get_quandl_data('ECB/EURHKD')

        elif from_cur == self.USD and to_cur == self.HKD:
            rate_info = self.get_quandl_data('FRED/DEXHKUS')

        elif from_cur == self.AUD and to_cur == self.HKD:
            rate_info = self.get_quandl_data('RBA/FXRHKD')

        return rate_info

    def get_quandl_data(self, query_info, start_date=None, end_date=None, data_dict=True, transform=None):
        if start_date is None:
            start_date = self.get_start_date()

        if end_date is None:
            end_date = self.get_end_date()
        quandl_data = quandl.get(query_info, start_date=start_date, end_date=end_date, returns='numpy',
                                 transform=transform)
        if not data_dict:
            return quandl_data
        data_dict = {}
        for date, rate in quandl_data:
            data_dict[date.strftime("%Y-%m-%d")] = rate
        return data_dict

    def get_interest_rate(self, required_info):

        # load interest rate file form data file
        if self._date_list is None:
            self.generate_date_list()
        raw_data = load_interest_rate()
        data_list = {}
        for date in self._date_list:
            if date in raw_data and required_info in raw_data[date] and raw_data[date][required_info] is not None:
                data_list[date] = float(raw_data[date][required_info])
            else:
                print "data %s not in raw data" % date
                new_info = get_hk_interest_rate(date)
                temp_date = date
                while new_info is None or required_info not in new_info or new_info[required_info] is None:
                    temp_date = get_ahead_date(temp_date, 1)
                    new_info = get_hk_interest_rate(detail_date=temp_date)
                else:
                    data_list[date] = float(new_info[required_info])
                    if date not in raw_data:
                        raw_data[date] = new_info
                    else:
                        raw_data[date][required_info] = new_info[required_info]

        save_interest_rate(raw_data)
        return data_list


if __name__ == "__main__":
    test = FundamentalAnalysis()
    test.set_start_date("2006-04-14")
    test.set_end_date("2006-05-31")
    a = test.fundamental_analysis([test.US10Y_BOND], test.FA_RAW_DATA)
    import pprint

    pprint.pprint(a, width=150)
    # print test.fundamental_analysis([{test.GOLDEN_PRICE: True}], fa_type=test.FA_RAW_DATA)
