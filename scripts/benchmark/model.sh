#!/usr/bin/bash

port="21804"
GPUs="0,1,2,3,4,5,6,7"

# Taking mistralai/Mistral-7B-v0.1 as an example.
model_names=("mistral") # The model to be compressed.
drop_modules=("mlp" "attn" "block") # The modules to be dropped.
drop_nums=("4" "8") # The number of dropped modules.

tasks=("boolq" "rte" "openbookqa" "piqa" "mmlu" "winogrande" "gsm8k" "hellaswag" "arc_challenge")
num_fewshots=("0" "0" "0" "0" "5" "5" "5" "10" "25")

for model_name in "${model_names[@]}"
do
    # Download the model to a local directory. 
    git lfs install
    git clone https://huggingface.co/mistralai/Mistral-7B-v0.1
    mv Mistral-7B-v0.1 ./"$model_name"_model
done