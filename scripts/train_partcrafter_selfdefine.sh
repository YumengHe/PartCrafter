#!/usr/bin/env bash

NUM_MACHINES=${NUM_MACHINES:-1}
NUM_LOCAL_GPUS=${NUM_LOCAL_GPUS:-2}   # 默认两卡
MACHINE_RANK=${MACHINE_RANK:-0}

export WANDB_API_KEY="${WANDB_API_KEY:-}"       # 用 wandb 就填，不用可留空
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"  # 默认用 2,3 两卡

accelerate launch \
    --num_machines $NUM_MACHINES \
    --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
    --machine_rank $MACHINE_RANK \
    src/train_partcrafter.py \
        --config configs/mp8_nt512_selfdefine2.yaml \
        --output_dir output_partcrafter \
        --tag result_2gpu \
        --pin_memory \
        --allow_tf32 \
        --gradient_accumulation_steps 8 \
        --use_ema \
        --scale_lr \
        --num_workers 2 \
$@

# bash scripts/train_partcrafter_selfdefine.sh