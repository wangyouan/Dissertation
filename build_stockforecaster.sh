#!/usr/bin/env bash

echo "clear unused file"

for directory in "build" "stockforecaster.egg-info" "stockforecaster.egg-info" "dist"
do
    if [ -d ${directory} ]
    then
        echo "remove $directory"
        rm -rf ${directory}
    fi
done

#git pull origin dev
python package_forecaster.py bdist_egg

echo "Build success, clean processing files"

for directory in "build" "stockforecaster.egg-info"
do
    if [ -d ${directory} ]
    then
        echo "remove $directory"
        rm -rf ${directory}
    fi
done