export CUDA_VISIBLE_DEVICES=0
cd models/VMmabaGP/
conda activate vmamba_env
 python main.py --mode train  --dataname Dendrite_growth  --epoch 200 --ex_name den_preformer_test  --res_dir ../../work_dirs --data_root ../../data --lr 0.0001 
