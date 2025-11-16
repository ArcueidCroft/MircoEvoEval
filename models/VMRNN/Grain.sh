export CUDA_VISIBLE_DEVICES=0

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")

EX_NAME="Grain_growth"

python tools/train.py \
    --config_file configs/mmnist/VMRNN-D.py \
    --dataname Grain_growth \
    --batch_size 16 \
    --epoch 400 \
    --overwrite \
    --lr 1e-4 \
    --ex_name "$EX_NAME"
