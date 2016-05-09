#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: test_distributed_neural_network
# Author: Mark Wang
# Date: 9/5/2016

import os
import logging
import sys

from StockSimulator.RegressionMethod.distributed_neural_network import NeuralNetworkSpark
from StockSimulator.parse_data import DataParser
from StockSimulator.constant import LABEL_POINT
from StockSimulator import load_spark_context


def test_distributed_ann():
    data_file = os.path.join(os.path.abspath('data'), "0051.HK.csv")
    sc = load_spark_context("NeuralNetwork")[0]

    # logger = sc._jvm.org.apache.log4j
    # logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    # logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

    data = DataParser(path=data_file, window_size=3)
    data_list = data.load_data_from_yahoo_csv()
    close_train_data, close_test_data, open_train_data, open_test_data = \
        data.get_n_days_history_data(data_list, data_type=LABEL_POINT, normalized=True, spark_context=sc)

    neural = NeuralNetworkSpark([4, 5, 1], bias=1)
    model = neural.train(rdd_data=close_train_data, learn_rate=1e-3, error=1e-5, iteration=100, method=neural.BP_SGD)
    predict_result = close_test_data.map(lambda p: (p.label, DataParser.de_normalize(model.predict(p.features),
                                                                                     p.features))).cache()
    model.save_model("neural_network_model_0051")
    mse = DataParser.get_MSE(predict_result)
    print mse


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    test_distributed_ann()