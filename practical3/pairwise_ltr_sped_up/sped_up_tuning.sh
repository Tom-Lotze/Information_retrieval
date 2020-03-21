#!/bin/bash


echo "Starting gridsearch"

declare -a hidden_list=("128, 128, 128" "512, 128, 8" "256, 10" "256" "512, 512, 512, 512")
declare -a lr_list=("0.001" "0.01" "0.005" "0.05")

> pairwise_ltr_sped_up/gridsearch_results.txt

for h_u in "${hidden_list[@]}"
do
    for lr in "${lr_list[@]}"
    do
        echo "hidden units: $h_u, learning rate: $lr"
        python pairwise_ltr_sped_up/pairwise_ltr_sped_up.py --hidden_units "$h_u" --learning_rate "$lr" >> pairwise_ltr_sped_up/gridsearch_results.txt
    done
done

