#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: file_operation
# Author: Mark Wang
# Date: 13/6/2016

import pickle


def load_data_from_file(path):
    f = open(path)
    data = pickle.load(f)
    f.close()
    return data


def save_data_to_file(path, data):
    f = open(path, 'w')
    pickle.dump(data, f)
    f.close()