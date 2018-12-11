#!/bin/bash

# cd into data
work_dir=$(pwd)
data_dir=data/myconnectome
mkdir -p $data_dir
cd $data_dir

# fetch data
wget -N -r -l inf --no-remove-listing -nH --cut-dirs=3 http://web.stanford.edu/group/poldracklab/myconnectome-data/base

# cd to original working directory
cd ${work_dir}