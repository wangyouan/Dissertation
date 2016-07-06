#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: __init__.py
# Author: Mark Wang
# Date: 9/5/2016



start_date = "2011-01-06"
end_date = "2014-01-06"
test_ratio = "2013-01-06"

predict_list = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK', '0006.HK', '0011.HK', '0012.HK', '0014.HK',
                '0016.HK', '0017.HK', '0019.HK', '0023.HK', '0027.HK', '0031.HK', '0043.HK', '0064.HK', '0066.HK',
                '0120.HK', '0268.HK', '0291.HK', '0455.HK', '0471.HK', '0546.HK', '0577.HK', '0688.HK', '0700.HK',
                '0737.HK', '0745.HK', '0777.HK', '0845.HK', '0872.HK', '1051.HK', '1112.HK', '1117.HK', '1181.HK',
                '1230.HK', '1251.HK', '1314.HK', '1361.HK', '1613.HK', '1918.HK', '2005.HK', '2362.HK', '2383.HK',
                '2789.HK', '3777.HK', '6823.HK', '8050.HK', '8123.HK']

# predict_list = ['0902.HK', '0836.HK', '2282.HK', '0120.HK', '0315.HK']
# predict_list = ['6823.HK']


if __name__ == '__main__':
    from StockInference.util.get_history_stock_price import get_all_data_about_stock
    from select_stock_list import stock_list

    # predict_list.sort()
    # print predict_list
    new_predict_list = []

    for symbol in predict_list:
        data = get_all_data_about_stock(symbol, start_date=start_date, end_date=end_date, remove_zero_volume=True)
        # print len(data)
        if len(data) > 0.95 * 737:
            new_predict_list.append(symbol)
        elif len(data) == 0:
            print symbol

    predict_set = set(predict_list)

    for symbol in stock_list:
        if symbol not in predict_set:
            data = get_all_data_about_stock(symbol, start_date=start_date, end_date=end_date, remove_zero_volume=True)
            if len(data) > 0.95 * 737:
                new_predict_list.append(symbol)
                if len(new_predict_list) >= 50:
                    break

    print new_predict_list
