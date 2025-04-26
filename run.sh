#!/bin/bash

python get_activations.py \
    --dataset "DRC" \
    --model_dir "/root/autodl-tmp/models/Qwen1.5-14B-Chat" \
    --save_dir "./assets/Qwen1.5-14B-Chat_DRC"

python select_heads.py \
    --activations_path "./assets/Qwen1.5-14B-Chat_DRC/DRC_act.pt" \
    --model_dir "/root/autodl-tmp/models/Qwen1.5-14B-Chat" \
    --save_dir "./assets/Qwen1.5-14B-Chat_DRC"

python generate.py \
    --dataset "DRC" \
    --model_dir "/root/autodl-tmp/models/Qwen1.5-14B-Chat" \
    --activations_path "./assets/Qwen1.5-14B-Chat_DRC/DRC_act.pt" \
    --selected_heads_path "./assets/Qwen1.5-14B-Chat_DRC/selected_heads.json" \
    --save_dir "./assets/Qwen1.5-14B-Chat_DRC" \
    --rank 64 \
    --generation_method "baseline"
