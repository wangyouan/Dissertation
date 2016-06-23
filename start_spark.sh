#!/usr/bin/env bash
spark-submit --master "spark://Master:7077" \
    --py-files dist/StockInference-0.3-py2.7.egg \
    --driver-memory	1g \
    --executor-memory 2g \
    --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:./logs/log4j.properties" \
    src/test_combine_system.py

spark-submit --master "spark://Master:7077" \
    --py-files dist/StockInference-0.3-py2.7.egg \
    --driver-memory	1g \
    --executor-memory 2g \
    --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:./logs/log4j.properties" \
    src/stock_prediction.py