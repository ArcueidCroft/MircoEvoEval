export CUDA_VISIBLE_DEVICES=1

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")

EX_NAME="Spinodal_decomposition"

python tools/train.py \
    --config_file configs/mmnist/VMRNN-D.py \
    --dataname Spinodal_decomposition \
    --batch_size 16 \
    --epoch 400 \
    --overwrite \
    --lr 1e-4 \
    --ex_name "$EX_NAME"  \
    --find_unused_parameters
