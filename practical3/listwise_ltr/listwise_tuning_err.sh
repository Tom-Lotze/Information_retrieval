#!/bin/bash


echo "Starting gridsearch"

declare -a hidden_list=(128 256 512)
declare -a lr_list=("0.001" "0.01" "0.005" "0.05")

> listwise_ltr/json_files/gridsearch_results.txt

for h_u in "${hidden_list[@]}"
do
    for lr in "${lr_list[@]}"
    do
        echo "hidden units: $h_u, learning rate: $lr"
        python listwise_ltr/listwise_ltr.py --hidden_units "$h_u"\
         --learning_rate "$lr" --valid_each 1000 --metric 'ERR' --plot 1\
         >> listwise_ltr/json_files/gridsearch_results_err.txt
    done
done
