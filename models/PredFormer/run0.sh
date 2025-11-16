#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python tools/train.py -d Dendrite_growth -m PredFormer --epoch 200 -c configs/mmnist/PredFormer.py --ex_name Dendrite_growth