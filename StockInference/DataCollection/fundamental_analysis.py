#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: fundamental_analysis
# Author: Mark Wang
# Date: 1/6/2016

import numpy as np
import quandl

from StockInference.DataCollection.base_class import BaseClass
from StockInference.util.get_history_stock_price import get_all_data_about_stock
from StockInference.util.date_parser import get_ahead_date


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
        quandl.ApiConfig.api_key = "RYdPmBZoFyLXxg1RQ3fY"

    def _get_bond_price(self, symbol, ratio):
        bond_info = get_all_data_about_stock(symbol=symbol, start_date=get_ahead_date(self.get_start_date(), 5),
                                             end_date=self._true_end_date)
        bond_price_map = {}
        if not ratio:
            for i in bond_info:
                bond_price_map[i[0]] = float(i[4])

        else:

            bond_price_index = {}
            for i in range(len(bond_info)):
                bond_price_index[bond_info[i][0]] = i

            for date in self._date_list:
                if date not in bond_price_index:
                    bond_price_map[date] = 0
                else:
                    today_index = bond_price_index[date]
                    today_price = float(bond_info[today_index][4])
                    last_day_price = float(bond_info[today_index - 1][4])
                    bond_price_map[date] = (today_price - last_day_price) / last_day_price

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
            if not isinstance(info, dict) and info in self._bond_label_dict:
                bond_price = self._get_bond_price(self._bond_label_dict[info], ratio=ratio)

                calculated_info = self._merge_info(calculated_info=calculated_info, info_dict=bond_price)
            elif self.FROM in info:
                currency_exchange_rate = self._get_currency_exchange_rate(info[self.FROM], info[self.TO])
                calculated_info = self._merge_info(calculated_info=calculated_info, info_dict=currency_exchange_rate)
            elif self.GOLDEN_PRICE in info:
                golden_price = self._get_golden_price_in_cny(ratio=info.get(self.GOLDEN_PRICE, False))
                calculated_info = self._merge_info(calculated_info=calculated_info, info_dict=golden_price)

        return [i[1:] for i in calculated_info]

    def _get_golden_price_in_cny(self, ratio=False):
        if not ratio:
            return self.get_quandl_data("WGC/GOLD_DAILY_CNY")
        else:
            data = self.get_quandl_data("WGC/GOLD_DAILY_CNY", transform='rdiff')
            for date in data:
                data[date] *= 100
            return data

    def _get_currency_exchange_rate(self, from_cur, to_cur):
        rate_info = None
        if from_cur == self.EUR and to_cur == self.HKD:
            rate_info = self.get_quandl_data('ECB/EURHKD', transform='rdiff')

        elif from_cur == self.USD and to_cur == self.HKD:
            rate_info = self.get_quandl_data('FRED/DEXHKUS', transform='rdiff')

        elif from_cur == self.AUD and to_cur == self.HKD:
            rate_info = self.get_quandl_data('RBA/FXRHKD', transform='rdiff')

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
        import pkg_resources
        my_data = pkg_resources.resource_exists('StockInference', 'interest_data.dat')
        print my_data


if __name__ == "__main__":
    test = FundamentalAnalysis()
    test.set_start_date("2012-03-04")
    test.set_end_date("2013-03-04")
    test.get_interest_rate(None)
    # print test.fundamental_analysis([{test.GOLDEN_PRICE: True}], fa_type=test.FA_RAW_DATA)
