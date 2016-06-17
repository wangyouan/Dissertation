#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: CombineSystem
# Author: Mark Wang
# Date: 16/6/2016

from StockInference.inference_system import InferenceSystem


class MixInferenceSystem(InferenceSystem):
    """ Try to combine two training method together """

    def __init__(self, stock_symbol, features=None, direction_method=None, amount_method=None, output_file_path=None,
                 data_folder_path=None, model_path=None, using_exist_model=False, amount_type=None):
        InferenceSystem.__init__(self, stock_symbol=stock_symbol, data_folder_path=data_folder_path,
                                 features=features, output_file_path=output_file_path, model_path=model_path,
                                 using_exist_model=using_exist_model)
        if direction_method is None:
            self.trend_prediction_method = self.RANDOM_FOREST_CLASSIFIER
        else:
            self.trend_prediction_method = direction_method
        self.amount_prediction_method = amount_method
