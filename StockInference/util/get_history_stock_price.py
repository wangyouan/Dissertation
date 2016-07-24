#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: get_history_stock_price
# Author: Mark Wang
# Date: 29/5/2016





if __name__ == "__main__":
    # hsi_stock_symbol = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK', '0006.HK', '0011.HK', '0012.HK',
    #                     '0016.HK', '0017.HK', '0019.HK', '0023.HK', '0027.HK', '0066.HK', '0083.HK', '0101.HK',
    #                     '0135.HK', '0144.HK', '0151.HK', '0267.HK', '0293.HK', '0322.HK', '0386.HK', '0388.HK',
    #                     '0494.HK', '0688.HK', '0700.HK', '0762.HK', '0823.HK', '0836.HK', '0857.HK', '0883.HK',
    #                     '0939.HK', '0941.HK', '0992.HK', '1038.HK', '1044.HK', '1088.HK', '1109.HK', '1113.HK',
    #                     '1299.HK', '1398.HK', '1880.HK', '1928.HK', '2318.HK', '2319.HK', '2388.HK', '2628.HK',
    #                     '3328.HK', '3988.HK', '6823.HK']
    symbol_list = ['0291.HK']

    import datetime

    import pandas as pd
    from pandas.tseries.offsets import CustomBusinessDay

    from StockInference.util.hongkong_calendar import HongKongCalendar

    hk_cal = HongKongCalendar(start_year=1998, end_year=2016)
    hk_bd = CustomBusinessDay(calendar=hk_cal)

    for symbol in symbol_list:
        time.sleep(1)
        a = get_all_data_about_stock(symbol)
        start_time = datetime.datetime.strptime(a[0][0], '%Y-%m-%d')
        end_time = datetime.datetime.strptime(a[-1][0], '%Y-%m-%d')
        pd.DatetimeIndex(start=start_time, end=end_time, freq=hk_bd)
        print symbol, len(a), a[0][0], a[-1][0], pd.DatetimeIndex(start=start_time, end=end_time, freq=hk_bd).size

        # print get_all_data_about_stock('1113.HK')
