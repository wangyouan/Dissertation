#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: hongkong_calendar
# Author: Mark Wang
# Date: 7/6/2016

import urllib2

from BeautifulSoup import BeautifulSoup
from StockInference.util.holiday import AbstractHolidayCalendar, Holiday


class HongKongCalendar(AbstractHolidayCalendar):
    def __init__(self, start_year=2006, end_year=2016):
        AbstractHolidayCalendar.__init__(self, "Hong Kong")
        self.start_year = int(start_year)
        self.end_year = int(end_year)
        self.rules = []
        self.month_dict = {
            'Jan': 1,
            "Feb": 2,
            "Mar": 3,
            "Apr": 4,
            "May": 5,
            "Jun": 6,
            "Jul": 7,
            "Aug": 8,
            "Sep": 9,
            "Oct": 10,
            "Nov": 11,
            "Dec": 12
        }
        self._add_rules()

    def _add_rules(self):
        for year in range(self.start_year, self.end_year + 1):
            self._add_year_rules(year=year)

    def _add_year_rules(self, year):
        year_url = "http://www.timeanddate.com/calendar/?year={}&country=42".format(year)
        year_page = urllib2.urlopen(year_url)
        page_html = year_page.read()
        year_page.close()
        soup = BeautifulSoup(page_html)
        holiday_list = soup.find('table', id='ch1').find('table').find('table').findAll('tr')
        for holiday in holiday_list:
            detail_date = holiday('td')[0].text.split(' ')
            holiday_name = holiday('td')[1].text
            self.rules.append(Holiday(name=holiday_name, year=year, day=int(detail_date[0]),
                                      month=self.month_dict[detail_date[1]]))


if __name__ == "__main__":
    cal = HongKongCalendar(2000, 2017)
    import pickle

    with open('hk_cal.p', 'w') as f:
        pickle.dump(cal, f)
