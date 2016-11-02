#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: tensorflow_system
# Author: Mark Wang
# Date: 16/10/2016

import tensorflow as tf

from stockforecaster.constant import Constants


class TensorFlowTrainingSystem(Constants):
    def __init__(self, train_method, hidden_layer_num=None, rt_trees_num=None):
        self._train_method = train_method

    def train(self, features, label):
        pass

    def predict(self, features):
        pass

    def stop(self):
        pass
