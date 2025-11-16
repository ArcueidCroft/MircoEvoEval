export CUDA_VISIBLE_DEVICES=0
cd models/VMRNN/
conda activate VMRNN
 python tools/test.py  --dataname Dendrite_growth  --method VMRNN  --epoch 200 --config_file ../../config/VMRNN-D.py  --ex_name den  --res_dir ../../work_dirs --data_root ../../data --lr 0.0001 
