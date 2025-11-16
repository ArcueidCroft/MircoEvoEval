import os

def prepare_datasets(path, data_type, data_name):
    cmd = f'cp {path} '
    if data_type == 'grain':
        os.makedirs("data/Grain_growth", exist_ok=True)
        cmd = cmd + f' data/Grain_growth/{data_name}'
    elif data_type == 'plane':
        os.makedirs("data/Plane_wave_propagation", exist_ok=True)
        cmd = cmd + f' data/Plane_wave_propagation/{data_name}'
        
    elif data_type == 'spin':
        os.makedirs("data/Spinodal_decomposition", exist_ok=True)
        cmd = cmd + f' data/Spinodal_decomposition/{data_name}'
    elif data_type == 'den':
        os.makedirs("data/Dendrite_growth", exist_ok=True)
        cmd = cmd + f' data/Dendrite_growth/{data_name}'
    os.system(cmd)