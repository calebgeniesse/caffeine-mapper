#!/bin/bash

# get current work_dir
work_dir=$(pwd)
echo $work_dir

# mkdir + cd "data/myconnectome" (if not there already)
data_dir=$(dirname $0)/myconnectome
echo $data_dir

mkdir -p $data_dir
cd $data_dir
echo $(pwd)

# fetch data
wget -N -r -l inf --no-remove-listing -nH --cut-dirs=3 http://web.stanford.edu/group/poldracklab/myconnectome-data/base

# cd to original working directory
cd ${work_dir}
echo $(pwd)