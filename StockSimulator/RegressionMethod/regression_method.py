#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: regression_method
# Author: Mark Wang
# Date: 6/5/2016

import logging

from constants import Constants


class Regression(Constants):
    def __init__(self):
        self.weights = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def save_model(self, path):
        f = open(path, 'w')
        import pickle
        pickle.dump(self.weights, f)
        f.close()

    def load_model(self, path):
        f = open(path)
        import pickle
        self.weights = pickle.load(f)
        f.close()

    def get_loss(self, features, target, method=None):
        """
        Get the lose value of current target
        :param features: features of the output
        :param target: origin output
        :param method: method to measure the loss. Default method is "lse" (least square error)
        :return: the loss between feature and target
        """
        predict_value = self.predict(features)
        if method is None:
            method = self.LSE

        if method == self.LSE:
            return (target - predict_value) ** 2

    def predict(self, features):
        pass

    def train(self, rdd_data, learn_rate=None, iteration=None, error=None, method=None):
        pass
