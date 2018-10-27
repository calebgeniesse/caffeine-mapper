#!/bin/bash

# cd into data
wd=$(pwd)
mkdir -p data
cd data

# fetch data
wget -N -r -l inf --no-remove-listing -nH --cut-dirs=3 http://web.stanford.edu/group/poldracklab/myconnectome-data/base/combined_data_scrubbed

# cd to original working directory
cd ${wd}