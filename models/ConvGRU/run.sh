export CUDA_VISIBLE_DEVICES=0
cd models/ConvGRU/
conda activate OpenSTL
 python  main.py  --data_type den  --epochs 200 --res_dir ../../work_dirs --data_root ../../data --lr 0.0001 
