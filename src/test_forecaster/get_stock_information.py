#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertationbrowser = webdriver.Chrome("/Users/warn/chromedriver")

# File name: get_stock_information
# Author: Mark Wang
# Date: 10/11/2016

import pandas as pd
from selenium import webdriver

browser = webdriver.Chrome("/Users/warn/chromedriver")


def get_stock_profile(stock_symbol):
    url = 'http://finance.yahoo.com/quote/{}/profile'.format(stock_symbol)
    browser.get(url)

    company_name = browser.find_elements_by_xpath(
        '//*[@id="main-0-Quote-Proxy"]/section/div[2]/section/div/div[1]/div/h3')
    if len(company_name) == 0:
        return None, None, None

    company_name = company_name[0].text
    other_info = browser.find_elements_by_xpath(
        '//*[@id="main-0-Quote-Proxy"]/section/div[2]/section/div/div[1]/div/div/p[2]')
    sector_info = None
    industry = None

    if len(other_info) != 0:
        infos = other_info[0].text.split('\n')
        for info in infos:
            info_split = info.split(': ')
            if len(info_split) != 2:
                continue

            if info_split[0] == 'Sector':
                sector_info = info_split[1]
            elif info_split[0] == 'Industry':
                industry = info_split[1]

    return company_name, sector_info, industry


if __name__ == '__main__':
    hsi_stock_list = ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK', '0006.HK', '0011.HK', '0012.HK', '0016.HK',
                      '0017.HK', '0019.HK', '0023.HK', '0027.HK', '0066.HK', '0083.HK', '0101.HK', '0135.HK', '0144.HK',
                      '0151.HK', '0267.HK', '0293.HK', '0386.HK', '0388.HK', '0494.HK', '0688.HK', '0700.HK', '0762.HK',
                      '0823.HK', '0836.HK', '0857.HK', '0883.HK', '0939.HK', '0941.HK', '0992.HK', '1038.HK', '1044.HK',
                      '1088.HK', '1109.HK', '1299.HK', '1398.HK', '1880.HK', '1928.HK', '2018.HK', '2318.HK', '2319.HK',
                      '2388.HK', '2628.HK', '3328.HK', '3988.HK', '6823.HK', '^HSI']

    df = pd.DataFrame(columns=['Company Name', 'Sector', 'Industry'])
    df.index.name = 'Symbol'
    for stock in hsi_stock_list:
        name, sector, ind = get_stock_profile(stock)
        df.loc[stock] = {'Company Name': name,
                         'Sector': sector,
                         'Industry': ind}

    browser.close()
