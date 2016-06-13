#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: date_parser
# Author: Mark Wang
# Date: 10/6/2016

import datetime

from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

from StockInference.util.hongkong_calendar import HongKongCalendar

calendar = HongKongCalendar(start_year=2006, end_year=2016)
custom_business_day = CustomBusinessDay(calendar=calendar)
holidays = calendar.holidays()


def get_ahead_date(date_to_change, ahead_days):
    if isinstance(date_to_change, str):
        date_list = date_to_change.split('-')
        date_object = datetime.datetime(year=int(date_list[0]), month=int(date_list[1]), day=int(date_list[2]))
    else:
        date_object = date_to_change
    date_object -= ahead_days * custom_business_day
    return date_object.strftime("%Y-%m-%d")


def string_to_datetime(input_date):
    if isinstance(input_date, str):
        date_list = input_date.split('-')
        date_list = map(int, date_list)
        new_date = datetime.datetime(date_list[0], date_list[1], date_list[2])
        return new_date
    else:
        return input_date


def is_holiday(input_date):

    if isinstance(input_date, str):
        date_list = input_date.split('-')
        date_object = datetime.datetime(year=int(date_list[0]), month=int(date_list[1]), day=int(date_list[2]))
    else:
        date_object = input_date

    return date_object.weekday() in [5, 6] or date_object in holidays


def get_united_states_market_info(data_dict, date_list, date_bound):
    if isinstance(date_bound, str):
        date_bound = string_to_datetime(date_bound)
    price_info = {}
    us_day_offset = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    for date in date_list:
        date_time = string_to_datetime(date)
        us_date = date_time - us_day_offset
        while us_date.strftime("%Y-%m-%d") not in data_dict:
            us_date -= us_day_offset
            if us_date < date_bound:
                break
        if us_date.strftime("%Y-%m-%d") not in data_dict:
            price_info[date] = 0
        else:
            price_info[date] = data_dict[us_date.strftime("%Y-%m-%d")]

    return price_info