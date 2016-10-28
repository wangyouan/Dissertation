#!/usr/bin/env python
# -*- coding: utf-8 -*-

# File Name: easter
# Created by warn on 10/27/16

from datetime import date, datetime, timedelta
import numpy as np

from pandas.tseries.tools import normalize_date

# import after tools, dateutil check
from dateutil.easter import easter
import pandas.tslib as tslib
from pandas.tslib import Timestamp, OutOfBoundsDatetime

from pandas.tseries.offsets import Tick, DateOffset, Nano

import functools


def as_timestamp(obj):
    if isinstance(obj, Timestamp):
        return obj
    try:
        return Timestamp(obj)
    except (OutOfBoundsDatetime):
        pass
    return obj


def as_datetime(obj):
    f = getattr(obj, 'to_pydatetime', None)
    if f is not None:
        obj = f()
    return obj


def _is_normalized(dt):
    if (dt.hour != 0 or dt.minute != 0 or dt.second != 0 or
            dt.microsecond != 0 or getattr(dt, 'nanosecond', 0) != 0):
        return False
    return True


def apply_wraps(func):
    @functools.wraps(func)
    def wrapper(self, other):
        if other is tslib.NaT:
            return tslib.NaT
        elif isinstance(other, (timedelta, Tick, DateOffset)):
            # timedelta path
            return func(self, other)
        elif isinstance(other, (np.datetime64, datetime, date)):
            other = as_timestamp(other)

        tz = getattr(other, 'tzinfo', None)
        nano = getattr(other, 'nanosecond', 0)

        try:
            if self._adjust_dst and isinstance(other, Timestamp):
                other = other.tz_localize(None)

            result = func(self, other)
            if self._adjust_dst:
                result = tslib._localize_pydatetime(result, tz)

            result = Timestamp(result)
            if self.normalize:
                result = result.normalize()

            # nanosecond may be deleted depending on offset process
            if not self.normalize and nano != 0:
                if not isinstance(self, Nano) and result.nanosecond != nano:
                    if result.tz is not None:
                        # convert to UTC
                        value = tslib.tz_convert_single(
                            result.value, 'UTC', result.tz)
                    else:
                        value = result.value
                    result = Timestamp(value + nano)

            if tz is not None and result.tzinfo is None:
                result = tslib._localize_pydatetime(result, tz)

        except OutOfBoundsDatetime:
            result = func(self, as_datetime(other))

            if self.normalize:
                # normalize_date returns normal datetime
                result = normalize_date(result)

            if tz is not None and result.tzinfo is None:
                result = tslib._localize_pydatetime(result, tz)

        return result

    return wrapper


class Easter(DateOffset):
    """
    DateOffset for the Easter holiday using
    logic defined in dateutil.  Right now uses
    the revised method which is valid in years
    1583-4099.
    """
    _adjust_dst = True

    def __init__(self, n=1, **kwds):
        super(Easter, self).__init__(n, **kwds)

    @apply_wraps
    def apply(self, other):
        currentEaster = easter(other.year)
        currentEaster = datetime(
            currentEaster.year, currentEaster.month, currentEaster.day)
        currentEaster = tslib._localize_pydatetime(currentEaster, other.tzinfo)

        # NOTE: easter returns a datetime.date so we have to convert to type of
        # other
        if self.n >= 0:
            if other >= currentEaster:
                new = easter(other.year + self.n)
            else:
                new = easter(other.year + self.n - 1)
        else:
            if other > currentEaster:
                new = easter(other.year + self.n + 1)
            else:
                new = easter(other.year + self.n)

        new = datetime(new.year, new.month, new.day, other.hour,
                       other.minute, other.second, other.microsecond)
        return new

    def onOffset(self, dt):
        if self.normalize and not _is_normalized(dt):
            return False
        return date(dt.year, dt.month, dt.day) == easter(dt.year)
