import os
import argparse
import sys
import time
sys.path.append("../models/VMRNN/tools")
from .prepare_data import prepare_datasets
# from ..models.VMRNN.tools.train import start_train
def start_train(args):
    print("start train")
    
    method = args.method
    config_file = args.config_file
    data_type = args.data_type
    epoch = args.epoch
    lr = args.lr
    gpu = args.gpu
    ex_name = args.ex_name
    if args.test_file != None:
        prepare_datasets(args.test_file, data_type, 'test.npy')
    if args.train_file != None:
        prepare_datasets(args.train_file, data_type, 'trian.npy')
    if args.valid_file != None:
        prepare_datasets(args.valid_file, data_type, 'valid.npy')
            
    if method == "PredFormer":
        start_predformer(method, data_type, config_file, epoch, gpu, lr, ex_name)
    elif method == "ConvGRU":
        start_convgru(method, data_type, config_file, epoch, gpu, lr, ex_name)
    elif method == "VMamba":
        start_vmamba(method, data_type, config_file, epoch, gpu, lr, ex_name)
    elif method == "VMRNN":
        start_vmrnn(method, data_type, config_file, epoch, gpu, lr, ex_name)
    else:
        start_openstl(method, data_type, config_file, epoch, gpu, lr, ex_name)
        # start_train(args)

def start_openstl(method, data_type, config_file, epoch, gpu, lr, ex_name):
    gpu_cmd = f"export CUDA_VISIBLE_DEVICES={gpu}"
    pre = "cd models/OpenSTL/"
    env = "conda activate OpenSTL"
    cmd = " python tools/train.py "
    d = f" --dataname {data_type} "
    m = f" --method {method} "
    epoch_cmd = f" --epoch {epoch}"
    c = f" --config_file ../../{config_file} "
    save_dir = f" --ex_name {ex_name} "
    r = " --res_dir ../../work_dirs --data_root ../../data"
    cmd = cmd+d+m+epoch_cmd+c+save_dir + r
    if lr != None:
        cmd = cmd+f" --lr {lr} "
    path = "models/OpenSTL/run.sh"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(gpu_cmd +"\n")
        f.write(pre+ "\n")
        f.write(env+ "\n")
        f.write(cmd+ "\n")
    time.sleep(1)
    print("Creating Command!")
    
    os.system("bash models/OpenSTL/run.sh")
    print(f"Runing, the log file is in work_dirs/{ex_name}")
        
def start_vmrnn(method, data_type, config_file, epoch, gpu, lr, ex_name):
    if data_type == "spin":
        data_type = "Spinodal_decomposition"
    elif data_type == "grain":
        data_type = "Grain_growth"
    elif data_type == "plane":
        data_type = "Plane_wave_propagation"
    elif data_type == "den":
        data_type = "Dendrite_growth"

    gpu_cmd = f"export CUDA_VISIBLE_DEVICES={gpu}"
    pre = "cd models/VMRNN/"
    env = "conda activate VMRNN"
    cmd = " python tools/train.py "
    d = f" --dataname {data_type} "
    m = f" --method {method} "
    epoch_cmd = f" --epoch {epoch}"
    c = f" --config_file ../../{config_file} "
    save_dir = f" --ex_name {ex_name} "
    r = " --res_dir ../../work_dirs --data_root ../../data"
    cmd = cmd+d+m+epoch_cmd+c+save_dir + r
    if lr != None:
        cmd = cmd+f" --lr {lr} "
    path = "models/VMRNN/run.sh"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(gpu_cmd +"\n")
        f.write(pre+ "\n")
        f.write(env+ "\n")
        f.write(cmd+ "\n")
    time.sleep(1)
    print("Creating Command!")
    
    os.system("bash models/VMRNN/run.sh")
    # print(f"Runing, the log file is in wor)

def start_predformer(method, data_type, config_file, epoch, gpu, lr, ex_name):
    if data_type == "spin":
        data_type = "Spinodal_decomposition"
    elif data_type == "grain":
        data_type = "Grain_growth"
    elif data_type == "plane":
        data_type = "Plane_wave_propagation"
    elif data_type == "den":
        data_type = "Dendrite_growth"
    print("start_predformer")
    gpu_cmd = f"export CUDA_VISIBLE_DEVICES={gpu}"
    pre = "cd models/PredFormer/"
    env = "conda activate PredFormer"
    cmd = " python tools/train.py "
    d = f" --dataname {data_type} "
    m = f" --method {method} "
    epoch_cmd = f" --epoch {epoch}"
    c = f" --config_file ../../{config_file} "
    save_dir = f" --ex_name {ex_name} "
    r = " --res_dir ../../work_dirs --data_root ../../data"
    cmd = cmd+d+epoch_cmd+c+save_dir + r + " --opt adamw "
    if lr != None:
        cmd = cmd+f" --lr {lr} "
    path = "models/PredFormer/run.sh"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(gpu_cmd +"\n")
        f.write(pre+ "\n")
        f.write(env+ "\n")
        f.write(cmd+ "\n")
    time.sleep(1)
    print("Creating Command!")
    
    os.system("bash models/PredFormer/run.sh")

def  start_vmamba(method, data_type, config_file, epoch, gpu, lr, ex_name):
    if data_type == "spin":
        data_type = "Spinodal_decomposition"
    elif data_type == "grain":
        data_type = "Grain_growth"
    elif data_type == "plane":
        data_type = "Plane_wave_propagation"
    elif data_type == "den":
        data_type = "Dendrite_growth"
    
    gpu_cmd = f"export CUDA_VISIBLE_DEVICES={gpu}"
    pre = "cd models/VMmabaGP/"
    env = "conda activate vmamba_env"
    cmd = " python main.py --mode train "
    d = f" --dataname {data_type} "
    m = f" --method {method} "
    epoch_cmd = f" --epoch {epoch}"
    c = f" --config_file ../../{config_file} "
    save_dir = f" --ex_name {ex_name} "
    r = " --res_dir ../../work_dirs --data_root ../../data"
    cmd = cmd+d+epoch_cmd+save_dir + r
    if lr != None:
        cmd = cmd+f" --lr {lr} "
    path = "models/VMmabaGP/run.sh"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(gpu_cmd +"\n")
        f.write(pre+ "\n")
        f.write(env+ "\n")
        f.write(cmd+ "\n")
    time.sleep(1)
    print("Creating Command!")
    
    os.system("bash models/VMmabaGP/run.sh")
    
def start_convgru(method, data_type, config_file, epoch, gpu, lr, ex_name):
    gpu_cmd = f"export CUDA_VISIBLE_DEVICES={gpu}"
    pre = "cd models/ConvGRU/"
    env = "conda activate OpenSTL"
    cmd = " python  main.py "
    d = f" --data_type {data_type} "
    m = f" --method {method} "
    epoch_cmd = f" --epochs {epoch}"
    c = f" --config_file ../../{config_file} "
    save_dir = f" --ex_name {ex_name} "
    r = " --res_dir ../../work_dirs --data_root ../../data"
    cmd = cmd+d+epoch_cmd + r
    if lr != None:
        cmd = cmd+f" --lr {lr} "
    path = "models/ConvGRU/run.sh"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(gpu_cmd +"\n")
        f.write(pre+ "\n")
        f.write(env+ "\n")
        f.write(cmd+ "\n")
    time.sleep(1)
    print("Creating Command!")
    
    os.system("bash models/ConvGRU/run.sh")
    # print(f"Runing, the log file is in work_dirs/{ex_name}")

    
    
    