#!/usr/bin/env bash

echo "clear unused file"

for directory in "dist" "build" "output" "StockSimulator.egg-info"
do
    if [ -d $directory ]
    then
        echo "remove $directory"
        rm -rf $directory
    fi
done

if [ -d "data" ]
then
    echo "data already exists"
else
    python get_data/yahoo_api.py ./data/
fi

git reset --hard
git pull origin dev
python package.py bdist_egg
