#!/usr/bin/env bash
spark-submit --master "spark://Master:7077" \
    --py-files dist/StockInference-0.2-py2.7.egg \
    --driver-memory	1g \
    --executor-memory 2g \
    --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:./logs/log4j.properties" \
    StockInference/inference_system.py