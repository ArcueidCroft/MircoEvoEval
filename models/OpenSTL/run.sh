export CUDA_VISIBLE_DEVICES=0
cd models/OpenSTL/
conda activate OpenSTL
 python tools/test.py  --dataname den  --method SimVP  --epoch None --config_file ../../../../config/simvp/SimVP_gSTA.py  --ex_name Debug  --res_dir ../../work_dirs --data_root ../../data
