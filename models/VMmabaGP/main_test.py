import argparse
from exp import Exp
import time
import warnings
warnings.filterwarnings('ignore')


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=3, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='Dendrite_growth', choices=['Dendrite_growth', 'Grain_growth','Dendrite','taxibj','mmnist','p','Spinodal_decomposition','Plane_wave_propagation'])
    parser.add_argument('--model', default="VMambaGP", type=str, choices=['VMambaGP', 'SimVP','SimVP_Model'])

    # model parameters
    parser.add_argument('--in_shape', default=[10, 1, 64,64], type=int, nargs='*')  # [10, 1, 64, 64] for Grain_growth, [13, 1, 128, 128] for p

    parser.add_argument('--hid_S', default=64, type=int)
    parser.add_argument('--hid_T', default=256, type=int)
    parser.add_argument('--N_S', default=2, type=int)
    parser.add_argument('--N_T', default=6, type=int)
    parser.add_argument('--groups', default=4, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--pre_seq_length', default=10, type=int)
    parser.add_argument('--mode', default="test", type=str, choices=['train', 'test'])

    return parser


if __name__ == '__main__':
    start = time.time()
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    
    if args.mode == "train":
       print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> train <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    #    exp.train(args)
    elif args.mode == "test":
       print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
       _ = exp.test1() 
    end = time.time()
    print("Using time :%.2f"%(end-start))
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    # mse = exp.test(args)