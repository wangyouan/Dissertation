#!/usr/bin/env bash

NUM1=0

if [ $# -eq $NUM1 ]
then
    spark-submit --master "spark://Master:7077" \
        --py-files dist/StockInference-0.2-py2.7.egg \
        --driver-memory	1g \
        --executor-memory 2g \
        --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:./logs/log4j.properties" \
        src/predict_price.py
elif [ $# -eq 1 ]
then
    spark-submit --master "spark://Master:7077" \
        --py-files dist/StockInference-0.2-py2.7.egg \
        --driver-memory	1g \
        --executor-memory 2g \
        --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:./logs/log4j.properties" \
        src/predict_price.py $1
else
    spark-submit --master "spark://Master:7077" \
        --py-files dist/StockInference-0.2-py2.7.egg \
        --driver-memory	1g \
        --executor-memory 2g \
        --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:./logs/log4j.properties" \
        src/predict_price.py $1 $2
fi
