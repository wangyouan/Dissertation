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
from pandas.tseries.offsets import CustomBusinessDay

from StockInference.DataCollection.base_class import BaseClass
from StockInference.util.get_history_stock_price import get_all_data_about_stock
from StockInference.util.date_parser import get_ahead_date, get_united_states_market_info
from StockInference.util.get_hk_interest_rate import get_hk_interest_rate

data_file_name = "interest_rate.dat"


def load_interest_rate():
    path = os.path.join(os.path.dirname(__file__), data_file_name)
    if os.path.isfile(path):
        f = open(path)
        raw_data = pickle.load(f)
        f.close()
        return raw_data
    else:
        print ("cannot find data file location")
        return {}


def save_interest_rate(raw_data):
    try:
        f = open(os.path.join(os.path.dirname(__file__), data_file_name), 'w')
        pickle.dump(raw_data, f)
        f.close()
    except Exception, e:
        print "can not save data as %s" % e


class FundamentalAnalysis(BaseClass):
    def __init__(self, logger=None):
        BaseClass.__init__(self, logger)
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
        file_name = "bond_{}".format(symbol)

        bond_price_map = self.load_data_from_file(file_name)
        if bond_price_map is None:
            bond_info = get_all_data_about_stock(symbol=symbol, start_date=get_ahead_date(self.get_start_date(), 20),
                                                 end_date=self.get_true_end_date(), remove_zero_volume=False)
            bond_price_map = {}
            for i in bond_info:
                bond_price_map[i[0]] = float(i[4])
            if location == self.UNITED_STATES:
                bond_price_map = get_united_states_market_info(bond_price_map, self.get_date_list(),
                                                               get_ahead_date(self.get_start_date(), 10))

        self.save_data_to_file(file_name, bond_price_map)
        return bond_price_map

    def fundamental_analysis(self, required_info):
        return self.raw_fundamental_analysis(required_info=required_info)

    def raw_fundamental_analysis(self, required_info):
        date_list = self.get_date_list()
        calculated_info = [[i] for i in date_list]
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
                elif info == self.GOLDEN_PRICE:
                    golden_price = self._get_golden_price_in_cny()
                    calculated_info = self._merge_info(calculated_info=calculated_info, info_dict=golden_price)

            elif isinstance(info, dict):
                if self.FROM in info:
                    currency_exchange_rate = self._get_currency_exchange_rate(info[self.FROM], info[self.TO])
                    calculated_info = self._merge_info(calculated_info=calculated_info,
                                                       info_dict=currency_exchange_rate)

        return [i[1:] for i in calculated_info]

    def _get_golden_price_in_cny(self):
        file_name = "golden"
        data = self.load_data_from_file(file_name)
        if data is not None:
            return data
        else:
            data = self.get_quandl_data("WGC/GOLD_DAILY_CNY")
            self.save_data_to_file(file_name, data)
            return data

    def _get_currency_exchange_rate(self, from_cur, to_cur):
        file_name = "from_{}_to_{}".format(from_cur, to_cur)
        rate_info = self.load_data_from_file(file_name)
        if rate_info is not None:
            pass
        elif from_cur == self.EUR and to_cur == self.HKD:
            rate_info = self.get_quandl_data('ECB/EURHKD')
            self.save_data_to_file(file_name=file_name, data=rate_info)

        elif from_cur == self.USD and to_cur == self.HKD:
            rate_info = self.get_quandl_data('FRED/DEXHKUS', start_date=get_ahead_date(self.get_start_date(), 20))
            rate_info = get_united_states_market_info(rate_info, self.get_date_list(),
                                                      get_ahead_date(self.get_start_date(), 10))
            self.save_data_to_file(file_name=file_name, data=rate_info)

        elif from_cur == self.AUD and to_cur == self.HKD:
            rate_info = self.get_quandl_data('RBA/FXRHKD')
            self.save_data_to_file(file_name=file_name, data=rate_info)

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

        file_name = "interest_{}".format(required_info)
        data = self.load_data_from_file(file_name)
        if data is not None:
            return data

        # load interest rate file form data file
        date_list = self.get_date_list()
        raw_data = load_interest_rate()
        data_list = {}
        for date in date_list:
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
        self.save_data_to_file(file_name, data_list)
        return data_list


if __name__ == "__main__":
    test = FundamentalAnalysis()
    test.set_start_date("2006-04-14")
    test.set_end_date("2006-05-31")
    a = test.fundamental_analysis([test.US10Y_BOND])
    import pprint

    pprint.pprint(a, width=150)
    # print test.fundamental_analysis([{test.GOLDEN_PRICE: True}], fa_type=test.FA_RAW_DATA)
