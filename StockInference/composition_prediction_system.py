#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: CombineSystem
# Author: Mark Wang
# Date: 16/6/2016

from StockInference.inference_system import InferenceSystem


class MixInferenceSystem(InferenceSystem):
    """ Try to combine two training method together """
    def __init__(self, stock_symbol, data_folder_path=None, training_method=None, features=None, output_file_path=None,
                 model_path=None, using_exist_model=False):
        InferenceSystem.__init__(self, stock_symbol=stock_symbol, data_folder_path=None,
                                 features=None, output_file_path=None, model_path=None, using_exist_model=False)
