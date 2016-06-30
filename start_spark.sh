#!/usr/bin/env bash

start_date="2011-01-06"
end_date="2015-01-06"
test_ratio="2014-01-06"

rm -rf output/*

spark-submit --master "spark://Master:7077" \
    --py-files dist/StockInference-0.3-py2.7.egg,src/__init__.py \
    --driver-memory	1g \
    --executor-memory 2g \
    --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:./logs/log4j.properties" \
    src/test_combine_system.py $start_date $end_date $test_ratio

spark-submit --master "spark://Master:7077" \
    --py-files dist/StockInference-0.3-py2.7.egg,src/__init__.py \
    --driver-memory	1g \
    --executor-memory 2g \
    --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:./logs/log4j.properties" \
    src/stock_prediction.py $start_date $end_date $test_ratio


file_name=output_3_1_${start_date:0:4}_${end_date:0:4}.tar.gz

tar -zcf $file_name output/
mv $file_name ../output/