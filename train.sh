python main.py --use_mode test \
    --data_type den \
    --method  VMRNN \
    --epoch 200 \
    --lr 0.0001 \
    --config_file config/VMRNN-D.py \
    --ex_name den \
    --gpu 0 \
    --aft_seq_length 10 \