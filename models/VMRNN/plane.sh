export CUDA_VISIBLE_DEVICES=0

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")

EX_NAME="Plane_wave_propagation"

python models/VMRNN/tools/train.py \
    --config_file config/VMRNN-D.py \
    --dataname Plane_wave_propagation \
    --batch_size 16 \
    --epoch 400 \
    --overwrite \
    --lr 1e-4 \
    --ex_name "$EX_NAME"  \
    --find_unused_parameters 
