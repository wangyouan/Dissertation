#!/usr/bin/env bash
spark-submit --master "spark://student32-x1:7077" \
    --py-files dist/StockSimulator-0.1-py2.7.egg \
    --driver-memory	1g \
    --executor-memory 1g \
    --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:~/src/dl4j-spark-cdh5-examples/src/main/resources/log4j.properties" \
    src/test_distributed_neural_network.py