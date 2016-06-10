#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: date_parser
# Author: Mark Wang
# Date: 10/6/2016

import datetime

from pandas.tseries.offsets import CustomBusinessDay

from StockInference.util.hongkong_calendar import HongKongCalendar

custom_business_day = CustomBusinessDay(calendar=HongKongCalendar(start_year=2004, end_year=2016))


def get_ahead_date(date_to_change, ahead_days):
    date_list = date_to_change.split('-')
    date_object = datetime.datetime(year=int(date_list[0]), month=int(date_list[1]), day=int(date_list[2]))
    date_object -= ahead_days * custom_business_day
    return date_object.strftime("%Y-%m-%d")
