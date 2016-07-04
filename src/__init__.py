#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: __init__.py
# Author: Mark Wang
# Date: 9/5/2016



start_date = "2012-01-06"
end_date = "2015-01-06"
test_ratio = "2014-01-06"

predict_list = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK', '0006.HK', '0008.HK', '0009.HK', '0010.HK',
                '0011.HK', '0012.HK', '0012.HK', '0013.HK', '0014.HK', '0015.HK', '0015.HK', '0016.HK', '0017.HK',
                '0018.HK', '0019.HK', '0020.HK', '0023.HK', '0024.HK', '0025.HK', '0027.HK', '0027.HK', '0031.HK',
                '0032.HK', '0034.HK', '0035.HK', '0038.HK', '0039.HK', '0041.HK', '0042.HK', '0043.HK', '0044.HK',
                '0045.HK', '0052.HK', '0053.HK', '0054.HK', '0056.HK', '0057.HK', '0059.HK', '0062.HK', '0064.HK',
                '0065.HK', '0066.HK', '0088.HK', '0168.HK', '0688.HK', '0700.HK', '0888.HK', '1123.HK', '6823.HK']

# predict_list = ['0902.HK', '0836.HK', '2282.HK', '0120.HK', '0315.HK']
# predict_list = ['6823.HK']


if __name__ == '__main__':
    from StockInference.util.get_history_stock_price import get_all_data_about_stock

    predict_list.sort()
    print predict_list

    # for symbol in predict_list:
    #     data = get_all_data_about_stock(symbol, start_date=start_date, end_date=end_date, remove_zero_volume=True)
    #     if len(data) < 738 * 0.9:
    #         print symbol, len(data)
