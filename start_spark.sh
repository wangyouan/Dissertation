#!/usr/bin/env bash
spark-submit --master "spark://stduent32-x1:7077" --py-files dist/StockSimulator-0.1-py2.7.egg StockSimulator/RegressionMethod/distributed_neural_network.py