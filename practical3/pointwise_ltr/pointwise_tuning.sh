#!/bin/bash


echo "Starting gridsearch"

declare -a hidden_list=("128, 128, 128" "512, 128, 8" "256, 10" "256" "512, 512, 512, 512")
declare -a lr_list=("0.001" "0.01" "0.005" "0.05")

> pointwise_ltr/json_files/gridsearch_results.txt

for h_u in "${hidden_list[@]}"
do
    for lr in "${lr_list[@]}"
    do
        echo "hidden units: $h_u, learning rate: $lr"
        python pointwise_ltr/pointwise_ltr.py --hidden_units "$h_u"\
         --learning_rate "$lr" \
         >> pointwise_ltr/json_files/gridsearch_results.txt
    done
done

