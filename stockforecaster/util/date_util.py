#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: date_util
# Author: Mark Wang
# Date: 28/7/2016

import datetime


def str2datetime(date_str):
    return datetime.datetime.strptime(date_str, '%Y-%m-%d')