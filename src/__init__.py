#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: __init__.py
# Author: Mark Wang
# Date: 9/5/2016


start_date = "2013-01-06"
end_date = "2016-01-06"
test_ratio = "2015-01-06"

predict_list = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK', '0006.HK', '0011.HK', '0012.HK', '0016.HK',
                '0017.HK', '0019.HK', '0023.HK', '0027.HK', '0066.HK', '0083.HK', '0101.HK', '0135.HK', '0144.HK',
                '0151.HK', '0267.HK', '0291.HK', '0293.HK', '0322.HK', '0386.HK', '0388.HK', '0494.HK', '0688.HK',
                '0700.HK', '0762.HK', '0777.HK', '0823.HK', '0836.HK', '0857.HK', '0883.HK', '0939.HK', '0941.HK',
                '0992.HK', '1038.HK', '1044.HK', '1088.HK', '1109.HK', '1398.HK', '1880.HK', '2318.HK', '2319.HK',
                '2388.HK', '2628.HK', '3328.HK', '3988.HK', '6823.HK']

# predict_list = ['0902.HK', '0836.HK', '2282.HK', '0120.HK', '0315.HK']
# predict_list = ['6823.HK']


if __name__ == '__main__':
    import random

    from StockInference.util.get_history_stock_price import get_all_data_about_stock
    from select_stock_list import stock_list

    # predict_list.sort()
    # print predict_list
    new_predict_list = []

    for symbol in predict_list:
        data = get_all_data_about_stock(symbol, start_date=start_date, end_date=end_date, remove_zero_volume=True)
        # print len(data)
        old_data = get_all_data_about_stock(symbol, end_date=start_date, remove_zero_volume=True)
        if len(data) > 0.95 * 737 and len(old_data) > 60:
            new_predict_list.append(symbol)
        elif len(data) == 0:
            print symbol

    predict_set = set(predict_list)

    while len(new_predict_list) < 50:
        symbol = random.choice(stock_list)
        while symbol in predict_set or symbol in new_predict_list:
            symbol = random.choice(stock_list)
        data = get_all_data_about_stock(symbol, start_date=start_date, end_date=end_date, remove_zero_volume=True)
        old_data = get_all_data_about_stock(symbol, end_date=start_date, remove_zero_volume=True)
        if len(data) > 0.95 * 737 and len(old_data) > 60:
            new_predict_list.append(symbol)

    new_predict_list.sort()
    print new_predict_list
