import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="MicroEvoEvalFramework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    print("start")
    # 
    parser.add_argument("--use_mode", choices=["train", "evaluate", "eval", "test"], help="choose to train models or evaluate")
    
    parser.add_argument("--path_true", type=str, default=None, 
                       help="the path to the true data")
    parser.add_argument("--path_pred", type=str, default=None, 
                       help="the path to the pred data")
    
    parser.add_argument("--data_type", choices=["den", "grain", "plane", "spin", "Spinodal_decomposition"],
                       help="the data type")
# train_params
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--ex_name', '-ex', default='Debug', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    # method parameters
    parser.add_argument('--method', '-m', default='SimVP', type=str,
                        choices=['ConvLSTM',  'E3DLSTM', 
                                 'MAU',  'MIM', 'PredNet', 'PredRNN', 'predrnn', 'PredRNNpp', 'predrnnpp', 'PredRNNv2', 'predrnnv2',
                                 'SimVP', 'simvp', 'TAU', 'ConvGRU','VMamba',
                                'SwinLSTM','VMRNN','vmrnn_d', 'vmrnn_b','predfomer', 'PredFormer'],
                        help='Name of video prediction method to train (default: "SimVP")')
    parser.add_argument('--config_file', '-c', default='../../config/simvp/SimVP_gSTA.py', type=str,
                        help='Path to the default config file')
    parser.add_argument('--epoch', '-e', default=None, type=int, help='end epochs (default: 200)')
    parser.add_argument('--aft_seq_length', default=10, type=int)
    # Training parameters (scheduler)
    parser.add_argument('--lr', default=None, type=float, help='Learning rate (default: 1e-3)')
    parser.add_argument('--test_file', default=None, type=str)
    parser.add_argument('--valid_file', default=None, type=str) 
    parser.add_argument('--train_file', default=None, type=str)
    parser.add_argument('--save_results', default=None, type=str)
    return parser.parse_args()
    

def default_parser():
    default_values = {
        # Set-up parameters
        'device': 'cuda',
        'dist': False,
        'display_step': 10,
        'res_dir': 'work_dirs',
        'ex_name': 'Dendrite_growth',
        'use_gpu': True,
        'fp16': False,
        'torchscript': False,
        'seed': 42,
        'diff_seed': False,
        'fps': False,
        'empty_cache': True,
        'find_unused_parameters': False,
        'broadcast_buffers': True,
        'resume_from': None,
        'auto_resume': False,
        'test': False,
        'inference': False,
        'deterministic': False,
        'launcher': 'pytorch',
        'local_rank': 0,
        'port': 29500,
        # dataset parameters
        'batch_size': 16,
        'val_batch_size': 16,
        'num_workers': 4,
        'data_root': './data',
        'dataname': 'Dendrite_growth',
        'pre_seq_length': 10,
        'aft_seq_length': 50,
        'total_length': 60,
        'use_augment': False,
        'use_prefetcher': False,
        'drop_last': False,
        # method parameters
        'method': 'SimVP',
        'config_file': 'configs/mmnist/VMRNN-D.py',
        'model_type': 'gSTA',
        'drop': 0,
        'drop_path': 0,
        'overwrite': False,
        # Training parameters (optimizer)
        'epoch': 200,
        'log_step': 1,
        'opt': 'adam',
        'opt_eps': None,
        'opt_betas': None,
        'momentum': 0.9,
        'weight_decay': 0,
        'clip_grad': None,
        'clip_mode': 'norm',
        'early_stop_epoch': -1,
        'no_display_method_info': False,
        # Training parameters (scheduler)
        'sched': 'onecycle',
        'lr': 1e-3,
        'lr_k_decay': 1.0,
        'warmup_lr': 1e-5,
        'min_lr': 1e-6,
        'final_div_factor': 1e4,
        'warmup_epoch': 0,
        'decay_epoch': 100,
        'decay_rate': 0.1,
        'filter_bias_and_bn': False,
    }
    return default_values