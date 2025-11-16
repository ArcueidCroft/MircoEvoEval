#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}
export PORT=29004
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=4,5

GPUS=2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="Dendrite_growth"

nohup $PYTHON -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py --dist \
    -d Dendrite_growth \
    --config_file configs/mmnist/VMRNN-D.py \
    --ex_name $EX_NAME \
    --batch_size 16 \
    --epoch 400 \
    --overwrite \
    --fps \
    --lr 1e-4 \
    --find_unused_parameters \
    --seed 42 --launcher="pytorch" >> logs/${EX_NAME}_run.log 2>&1 &
