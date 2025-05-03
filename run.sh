#!/bin/bash

dataset="DRC"
# model_dir="/root/autodl-tmp/models/Qwen3-8B"
model_dir="/root/autodl-tmp/models/Qwen1.5-14B-Chat"
assets_dir="/root/autodl-tmp/stylle_assets/Qwen1.5-14B-Chat_DRC"

python get_activations.py \
    --dataset $dataset \
    --model_dir $model_dir \
    --save_dir $assets_dir

python select_heads.py \
    --activations_path $assets_dir/act.pt \
    --model_dir $model_dir \
    --save_dir $assets_dir

python generate.py \
    --dataset $dataset \
    --model_dir $model_dir \
    --activations_path $assets_dir/act.pt \
    --selected_heads_path $assets_dir/selected_heads.json \
    --save_dir $assets_dir/R64 \
    --rank 64 \
    --generation_method "baseline"

python generate_ref.py \
    --dataset $dataset \
    --model_dir $model_dir \
    --save_dir $assets_dir/ref
