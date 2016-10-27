#!/usr/bin/env python
# -*- coding: utf-8 -*-

# File Name: easter
# Created by warn on 10/27/16

from pandas.tseries.offsets import DateOffset, apply_wraps, easter, datetime, tslib, date, _is_normalized


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
