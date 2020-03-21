#!/bin/bash


echo "Starting gridsearch"

declare -a hidden_list=(128 256 512)
declare -a lr_list=("0.001" "0.01" "0.005" "0.05")

> pairwise_ltr_sped_up/json_files/gridsearch_results.txt

for h_u in "${hidden_list[@]}"
do
    for lr in "${lr_list[@]}"
    do
        echo "hidden units: $h_u, learning rate: $lr"
        python pairwise_ltr_sped_up/pairwise_ltr_sped_up.py --hidden_units "$h_u" --learning_rate "$lr" --plot 1
         >> pairwise_ltr_sped_up/json_files/gridsearch_results.txt
    done
done
