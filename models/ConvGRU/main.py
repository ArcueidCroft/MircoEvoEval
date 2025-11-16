#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
# from datasets.mm import Dataloader
# from datasets.mm import Plane , Grain
from datasets.mm import OriginalDataloader
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import time
import pdb


try:
    import lpips
    from skimage.metrics import structural_similarity as cal_ssim
except:
    lpips = None
    cal_ssim = None


#TIMESTAMP = "Grain"
#TIMESTAMP = "Dendrite_growth"
#TIMESTAMP = "Plane_wave_propagation"
# TIMESTAMP = "Spindol"
parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('--data_type', default='spin', choices=['den', 'spin', 'grain', 'plane'])
parser.add_argument('--res_dir', default='../../work_dirs/', type=str)
parser.add_argument('--data_root', default='../../data' , type=str)
parser.add_argument('-cgru',
                    '--convgru',
                    help='use convgru as base cell',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=4,
                    type=int,
                    help='mini-batch size')
parser.add_argument('--lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('--frames_input',
                    default=10,
                    type=int,
                    help='sum of input frames')
parser.add_argument('--frames_output',
                    default=10,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('--epochs', default=400, type=int, help='sum of epochs')
args = parser.parse_args()

random_seed = 1996
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
data_root = args.data_root
save_dir = os.path.join(args.res_dir,args.data_type)
result   = args.res_dir  
trainFolder = []
validFolder = []
testFolder = []
if args.data_type == 'spin':
    trainFolder = OriginalDataloader(data_path=os.path.join(data_root, 'Spinodal_decomposition/train.npy'),
                            n_frames_input=args.frames_input,
                            n_frames_output=args.frames_output)
                            
    validFolder = OriginalDataloader(data_path=os.path.join(data_root, 'Spinodal_decomposition/valid.npy'),
                            n_frames_input=args.frames_input,
                            n_frames_output=args.frames_output)
                        ## dataset： Dendrite_growth ,Grain_growth, Plane_wave_propagation , Spinodal_decomposition   
    testFolder = OriginalDataloader(data_path=os.path.join(data_root, 'Spinodal_decomposition/test.npy'),   
                            n_frames_input=args.frames_input,
                            n_frames_output=args.frames_output)

elif args.data_type == 'grian':
    trainFolder = OriginalDataloader(data_path=os.path.join(data_root, 'Grain_growth/train.npy'),
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output)
                          
    validFolder = OriginalDataloader(data_path=os.path.join(data_root, 'Grain_growth/valid.npy'),
                            n_frames_input=args.frames_input,
                            n_frames_output=args.frames_output)
    testFolder = OriginalDataloader(data_path=os.path.join(data_root, 'Grain_growth/test.npy'),   
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output)
    
elif args.data_type == 'den':
    trainFolder = OriginalDataloader(data_path=os.path.join(data_root, 'Dendrite_growth/train.npy'),
                            n_frames_input=args.frames_input,
                            n_frames_output=args.frames_output)
                            
    validFolder = OriginalDataloader(data_path=os.path.join(data_root, 'Dendrite_growth/valid.npy'),
                            n_frames_input=args.frames_input,
                            n_frames_output=args.frames_output)
    testFolder = OriginalDataloader(data_path=os.path.join(data_root, 'Dendrite_growth/test.npy'),   
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output)
elif args.data_type == 'plane':
    trainFolder = OriginalDataloader(data_path=os.path.join(data_root, 'Plane_wave_propagation/train.npy'),
                            n_frames_input=args.frames_input,
                            n_frames_output=args.frames_output)
                            
    validFolder = OriginalDataloader(data_path=os.path.join(data_root, 'Plane_wave_propagation/valid.npy'),
                            n_frames_input=args.frames_input,
                            n_frames_output=args.frames_output)
                        ## dataset： Dendrite_growth ,Grain_growth, Plane_wave_propagation , Spinodal_decomposition   
    testFolder = OriginalDataloader(data_path=os.path.join(data_root, 'Plane_wave_propagation/test.npy'),   
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output)

trainLoader = torch.utils.data.DataLoader(trainFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False)
validLoader = torch.utils.data.DataLoader(validFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False)
                                          
testLoader = torch.utils.data.DataLoader(testFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False)

if args.convlstm:
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params
if args.convgru:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
else:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params


def MAE(pred, true, spatial_norm=True):
    if not spatial_norm:
        return np.mean(np.abs(pred-true), axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean(np.abs(pred-true) / norm, axis=(0, 1)).sum()


def MSE(pred, true, spatial_norm=True):
    if not spatial_norm:
        return np.mean((pred-true)**2, axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred-true)**2 / norm, axis=(0, 1)).sum()


def RMSE(pred, true, spatial_norm=True):
    if not spatial_norm:
        return np.sqrt(np.mean((pred-true)**2, axis=(0, 1)).sum())
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred-true)**2 / norm, axis=(0, 1)).sum())


def SSIM(pred, true, **kwargs):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = pred.astype(np.float64)
    img2 = true.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def train():
    '''
    main function to run the training
    '''
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    net = ED(encoder, decoder)
    run_dir = '../../runs/' + args.data_type
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoin.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0
    lossfunction = nn.MSELoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=4)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf
    for epoch in range(cur_epoch, args.epochs + 1):
        ###################
        # train the model #
        ###################
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (idx, targetVar, inputVar) in enumerate(t):
            inputs = inputVar.to(device)  # B,S,C,H,W
            label = targetVar.to(device)  # B,S,C,H,W
            optimizer.zero_grad()
            net.train()
            pred = net(inputs)  # B,S,C,H,W
            loss = lossfunction(pred, label)
            loss_aver = loss.item() / args.batch_size
            train_losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
        tb.add_scalar('TrainLoss', loss_aver, epoch)
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (idx, targetVar, inputVar) in enumerate(t):
                if i == 3000:
                    break
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                pred = net(inputs)
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / args.batch_size
                # record validation loss
                valid_losses.append(loss_aver)
                #print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })

        tb.add_scalar('ValidLoss', loss_aver, epoch)
        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(args.epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)
            
            
            
            
            
def test():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(encoder_params[0], encoder_params[1])
    decoder = Decoder(decoder_params[0], decoder_params[1])
    net = ED(encoder, decoder).to(device)
    #net = nn.DataParallel(net).cuda()
    ## Den:  checkpoint_62_0.000002.pth.tar , Grain: checkpoint_61_0.000286.pth.tar  ,Plane:checkpoint_107_0.000004.pth.tar , Spindol: checkpoint_188_0.000037.pth.tar 
    checkpoint_path = os.path.join(save_dir, 'checkpoint_188_0.000037.pth.tar')
    assert os.path.exists(checkpoint_path), f"Model not found: {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    lossfunction = nn.MSELoss().cuda()
    total_loss = 0
    all_preds = []  
    all_labels = []  

    with torch.no_grad():
        for _, targetVar, inputVar in tqdm(testLoader, desc="Testing"):
            inputs = inputVar.cuda()  # B, T_in, C, H, W
            labels = targetVar.cuda()  # B, T_out, C, H, W
            # pdb.set_trace()
            
            sample = 9
            output = []
            for i in range(sample):
            
              preds = net(inputs)  # B, T_out, C, H, W
              output.append(preds)
              inputs = preds
            output = torch.concat(output, dim =1 )
            
            
            loss = lossfunction(output, labels)
            total_loss += loss.item()

            all_preds.append(output.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

  
    all_preds = np.concatenate(all_preds, axis=0)  # [N, T_out, C, H, W]
    all_labels = np.concatenate(all_labels, axis=0)

 
    np.save(os.path.join(save_dir, 'predictions_90.npy'), all_preds)
    np.save(os.path.join(save_dir, 'ground_truth_90.npy'), all_labels)
    mse = MSE(all_preds, all_labels)
    mae = MAE(all_preds, all_labels)
    rmse = RMSE(all_preds, all_labels)
    ssim = 0
    for b in range(all_preds.shape[0]):
        for f in range(all_preds.shape[1]):
            ssim += cal_ssim(all_preds[b, f].squeeze(0),
                             all_labels[b, f].squeeze(0), multichannel=True, data_range=1.0)
    out_ssim = ssim / (all_preds.shape[0] * all_preds.shape[1])


    #avg_loss = total_loss / len(testLoader)
    #rmse = np.sqrt(avg_loss)
    with open(os.path.join(save_dir, 'test_results_90.txt'), 'w') as f:
        f.write(f"Test MSE Loss: {mse:.9f}\n")
        f.write(f"Test MAE Loss: {mae:.9f}\n")
        f.write(f"Test RMSE Loss: {rmse:.9f}\n")
        f.write(f"Test SSIM Loss: {out_ssim:.9f}\n")
    
    
    print(f"Test MSE Loss: {mse:.9f}, Test MAE Loss, {mae:.9f}, Test rmse Loss : {rmse:.9f} , Test SSIM Loss: {out_ssim:.9f}")



if __name__ == "__main__":
    train()
    start = time.time()
    test()
    end = time.time()
    print("Using time :%.2f"%(end-start))
