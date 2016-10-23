#!/usr/bin/env bash

WORKER_NUM=5
PACKAGE_NAME=stockforecaster-0.1-py2.7.egg

spark-submit --master local[${WORKER_NUM}] \
    --py-files dist/${PACKAGE_NAME} \
    --driver-memory	1g \
    --executor-memory 1g \
    src/test_forecaster/test_ann_classifier.py