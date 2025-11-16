export CUDA_VISIBLE_DEVICES=0
cd models/PredFormer/
conda activate PredFormer
 python tools/test.py  --dataname Dendrite_growth  --epoch 200 --config_file ../../config/PredFormer.py  --ex_name den_preformer_test  --res_dir ../../work_dirs --data_root ../../data --opt adamw  --lr 0.0001 
