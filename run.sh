#!/bin/bash

dataset=$1
model=$2
assets_dir=$3

# default values
dataset="DRC"
model="/root/autodl-tmp/models/Qwen1.5-14B-Chat"
assets_dir="/root/autodl-tmp/stylle_assets/Qwen1.5-14B-Chat_DRC"

if [ -n "$1" ]; then
    dataset=$1
fi

if [ -n "$2" ]; then
    model=$2
fi

if [ -n "$3" ]; then
    assets_dir=$3
fi


python get_activations.py \
    --dataset $dataset \
    --model $model \
    --save_dir $assets_dir

python select_heads.py \
    --activations_path $assets_dir/act.pt \
    --model $model \
    --save_dir $assets_dir

python generate.py \
    --dataset $dataset \
    --model $model \
    --activations_path $assets_dir/act.pt \
    --selected_heads_path $assets_dir/selected_heads.json \
    --save_dir $assets_dir/R64 \
    --rank 64 \
    --generation_method "baseline"

python generate_ref.py \
    --dataset $dataset \
    --model $model \
    --save_dir $assets_dir/ref
