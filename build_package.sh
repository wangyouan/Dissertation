#!/usr/bin/env bash

echo "clear unused file"

for directory in "build" "StockSimulator.egg-info" "StockInference.egg-info" "dist"
do
    if [ -d $directory ]
    then
        echo "remove $directory"
        rm -rf $directory
    fi
done

#if [ -d "data" ]
#then
#    echo "data already exists"
#else
#    python get_data/yahoo_api.py ./data/
#fi

#git pull origin dev
python package.py bdist_egg

echo "Build success, clean processing files"

for directory in "build" "StockInference.egg-info"
do
    if [ -d $directory ]
    then
        echo "remove $directory"
        rm -rf $directory
    fi
done