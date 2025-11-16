import os.path as osp
import json
import pickle

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import SimVP, VMambaGP, SimVP_Model
from tqdm import tqdm
from API import *
from utils import *
from dataloader import get_loader_segment
import pdb

class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.model + '_' + self.args.dataname)
        check_dir(self.path)

        self.checkpoints_path = self.path
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()
    def adjust_learning_rate(self,optimizer, epoch):
        lr = self.args.lr * (10-(epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _build_model(self):
        args = self.args
        if self.args.model == 'SimVP_Model':
            self.model = SimVP_Model(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model1 = SimVP_Model(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model2 = SimVP_Model(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model3 = SimVP_Model(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model4 = SimVP_Model(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model5 = SimVP_Model(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model6 = SimVP_Model(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model7 = SimVP_Model(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
        elif self.args.model == 'VMambaGP':
            self.model = VMambaGP(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model1 = VMambaGP(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model2 = VMambaGP(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model3 = VMambaGP(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model4 = VMambaGP(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model5 = VMambaGP(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model6 = VMambaGP(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model7 = VMambaGP(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            

        elif self.args.model == 'SimVP':
            self.model = SimVP(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model1 = SimVP(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model2 = SimVP(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model3 = SimVP(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model4 = SimVP(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model5 = SimVP(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model6 = SimVP(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            self.model7 = SimVP(tuple(args.in_shape), args.hid_S, args.hid_T, args.N_S, args.N_T).to(self.device)
            
            

    def _get_data(self):
        config = self.args.__dict__

        if self.args.dataname in ('taxibj', 'mmnist'):
            self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)
            self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader
        else:
            self.train_loader = get_loader_segment(self.args.batch_size, pre_seq_length=self.args.pre_seq_length,
                                                   mode='train', dataset=self.args.dataname)
            self.vali_loader = get_loader_segment(self.args.batch_size, pre_seq_length=self.args.pre_seq_length,
                                                  mode='val', dataset=self.args.dataname)

            self.test_loader = get_loader_segment(self.args.batch_size, pre_seq_length=self.args.pre_seq_length,
                                                     mode='test',
                                                     dataset=self.args.dataname)
            #pdb.set_trace()

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        self.optimizer1 = torch.optim.Adam(
            self.model1.parameters(), lr=self.args.lr)
        self.scheduler1 = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer1, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        self.optimizer2 = torch.optim.Adam(
           self.model2.parameters(), lr=self.args.lr)
        self.scheduler2 = torch.optim.lr_scheduler.OneCycleLR(
           self.optimizer2, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        self.optimizer3 = torch.optim.Adam(
           self.model3.parameters(), lr=self.args.lr)
        self.scheduler3 = torch.optim.lr_scheduler.OneCycleLR(
           self.optimizer3, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer,self.optimizer1
    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()

    def _save(self, name='1'):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, self.args.model + '_' + self.args.dataname + name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, self.args.model + '_' + self.args.dataname + name + '.pkl'), 'wb')
        pickle.dump(state, fw)
    def diff_div_reg(self, pred_y, batch_y, tau=0.1, eps=1e-12):
        B, T, C = pred_y.shape[:3]
        if T <= 2:  return 0
        gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
        gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
        softmax_gap_p = F.softmax(gap_pred_y / tau, -1)
        softmax_gap_b = F.softmax(gap_batch_y / tau, -1)
        loss_gap = softmax_gap_p * \
            torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
        return loss_gap.mean()

    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)
        recorder1 = Recorder(verbose=True)
        recorder2 = Recorder(verbose=True)
        recorder3 = Recorder(verbose=True)

        for epoch in range(config['epochs']):
            

            train_pbar = tqdm(self.train_loader)

            if self.args.dataname == 'p':
                train_loss = []
                self.model.train()
                self.model1.train()
                for batch_2_4, batch_4_6,batch_6_8 in train_pbar:
                    self.optimizer.zero_grad()
                    self.optimizer1.zero_grad()

                    loss2 = 0

                    batch_2_4, batch_4_6,batch_6_8=  batch_2_4.to(self.device, dtype=torch.float), batch_4_6.to(self.device, dtype=torch.float),batch_6_8.to(self.device, dtype=torch.float)

                    

                    pred_4_6 = self.model(batch_2_4)
                    pred_6_8 = self.model1(batch_4_6)
                    
                    if self.args.model=="VMambaGP" or self.args.model=="SimVP_Model":
                        loss = self.criterion(pred_4_6, batch_4_6)+0.1*self.diff_div_reg(pred_4_6, batch_4_6)
                        loss1 = self.criterion(pred_6_8, batch_6_8)+0.1*self.diff_div_reg(pred_6_8, batch_6_8)
                    else:
                        loss = self.criterion(pred_4_6, batch_4_6)
                        loss1 = self.criterion(pred_6_8, batch_6_8)

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    loss2 += loss
                    
                    
                    loss1.backward()
                    self.optimizer1.step()
                    self.scheduler1.step()
                    loss2 += loss1
                    
                    train_loss.append(loss2.item())
                train_loss = np.average(train_loss)
                with torch.no_grad():
                    vali_loss,vali_loss1 = self.vali(self.vali_loader)
                print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f} Learning_rate：{3:.4f}\n".format(epoch + 1,train_loss,vali_loss+vali_loss1,self.optimizer.param_groups[0]['lr']))
                recorder(vali_loss, self.model, self.path, name='')
                recorder1(vali_loss1, self.model1, self.path, name='1')

                self.test(self.test_loader)
            

            elif  self.args.dataname == 'Dendrite' :
                train_loss = []
                self.model.train()
                self.model1.train()
                self.model2.train()
                self.model3.train()
                for batch_0_1, batch_1_2,batch_2_3,batch_3_4,batch_4_5 in train_pbar:
                    loss=0
                    self.optimizer.zero_grad()
                    self.optimizer1.zero_grad()
                    self.optimizer2.zero_grad()
                    self.optimizer3.zero_grad()
                    batch_0_1, batch_1_2,batch_2_3,batch_3_4,batch_4_5 = batch_0_1.to(self.device, dtype=torch.float),batch_1_2.to(self.device, dtype=torch.float),batch_2_3.to(self.device, dtype=torch.float),batch_3_4.to(self.device, dtype=torch.float),batch_4_5.to(self.device, dtype=torch.float)
                    pred_1_2 = self.model(batch_0_1)            
                    pred_2_3 = self.model1(batch_1_2)                    
                    pred_3_4 = self.model2(batch_2_3)                   
                    pred_4_5 = self.model3(batch_3_4)
                    
                    if self.args.model=="VMambaGP":
                        loss = self.criterion(pred_1_2, batch_1_2)+0.1*self.diff_div_reg(pred_1_2, batch_1_2)
                        loss1 = self.criterion(pred_2_3, batch_2_3)+0.1*self.diff_div_reg(pred_2_3, batch_2_3)
                        loss2 = self.criterion(pred_3_4, batch_3_4)+0.1*self.diff_div_reg(pred_3_4, batch_3_4)
                        loss3 = self.criterion(pred_4_5, batch_4_5)+0.1*self.diff_div_reg(pred_4_5, batch_4_5)
                    else:
                        loss = self.criterion(pred_1_2, batch_1_2)
                        loss1 = self.criterion(pred_2_3, batch_2_3)
                        loss2 = self.criterion(pred_3_4, batch_3_4)
                        loss3 = self.criterion(pred_4_5, batch_4_5)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    loss1.backward()
                    self.optimizer1.step()
                    self.scheduler1.step()
                    
                    
                    loss2.backward()
                    self.optimizer2.step()
                    self.scheduler2.step()
                    
                    
                    loss3.backward()
                    self.optimizer3.step()
                    self.scheduler3.step()
                    
                    train_loss.append(loss.item())
                train_loss = np.average(train_loss)
                with torch.no_grad():
                    vali_loss,vali_loss1,vali_loss2,vali_loss3 = self.vali(self.vali_loader)
                print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f} Learning_rate：{3:.4f}\n".format(epoch + 1,train_loss,vali_loss+vali_loss1+vali_loss2+vali_loss3,self.optimizer.param_groups[0]['lr']))
                recorder(vali_loss, self.model, self.path, name='')
                recorder1(vali_loss1, self.model1, self.path, name='1')
                recorder2(vali_loss2, self.model2, self.path, name='2')
                recorder3(vali_loss3, self.model3, self.path, name='3')
                with torch.no_grad():
                    self.test(self.test_loader)
                                                       
                                
                
            elif self.args.dataname == 'Grain_growth' or self.args.dataname =='Spinodal_decomposition' or self.args.dataname =='Dendrite_growth' or self.args.dataname =='Plane_wave_propagation' :
                train_loss = []
                self.model.train()
                for batch_0_1, batch_1_2 in train_pbar:
                    
                    self.optimizer.zero_grad()
                    batch_0_1, batch_1_2 = batch_0_1.to(self.device, dtype=torch.float), batch_1_2.to(self.device, dtype=torch.float)
                    
                    pred_1_2 = self.model(batch_0_1)
                    if self.args.model == "VMambaGP":
                        loss = self.criterion(pred_1_2, batch_1_2) +0.1*self.diff_div_reg(pred_1_2, batch_1_2)
                    else:
                        loss = self.criterion(pred_1_2, batch_1_2)                                                       
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    train_loss.append(loss.item())
                train_loss = np.average(train_loss)
                with torch.no_grad():
                    vali_loss = self.vali(self.test_loader)
                print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f} Learning_rate：{3:.4f}\n".format(epoch + 1,train_loss,vali_loss,self.optimizer.param_groups[0]['lr']))
                recorder(vali_loss, self.model, self.path, name='')
                
                    
            elif self.args.dataname == 'mmnist':
                for batch_x, batch_y in train_pbar:
                    self.optimizer.zero_grad()
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    pred_y = self.model(batch_x)

                    loss = self.criterion(pred_y, batch_y)
                    train_loss.append(loss.item())
                    train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

        return self.model


    
    def vali(self, vali_loader):
        
        if self.args.dataname == 'p':
            self.model.eval()
            self.model1.eval()
     
            preds_lst1, trues_lst1 =[], []
            preds_lst2, trues_lst2 =[], []
            preds_lst3, trues_lst3 =[], []

            vali_pbar = tqdm(vali_loader)
            for i, ( batch_2_4, batch_4_6,batch_6_8) in enumerate(vali_pbar):
                
                batch_2_4, batch_4_6, batch_6_8=  batch_2_4.to(self.device, dtype=torch.float), batch_4_6.to(self.device, dtype=torch.float), batch_6_8.to(self.device, dtype=torch.float)
                
                pred_4_6 = self.model(batch_2_4)
                pred_6_8 = self.model1(batch_4_6)
                preds_lst1.append(pred_4_6.detach().cpu().numpy())
                trues_lst1.append(batch_4_6.detach().cpu().numpy())
                preds_lst2.append(pred_6_8.detach().cpu().numpy())
                trues_lst2.append(batch_6_8.detach().cpu().numpy())
               
                
            preds1 = np.concatenate(preds_lst1, axis=0)
            trues1 = np.concatenate(trues_lst1, axis=0)
            preds2 = np.concatenate(preds_lst2, axis=0)
            trues2 = np.concatenate(trues_lst2, axis=0)
            
            
            mse, mae, ssim1, psnr = metric(preds1, trues1, True)
            print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim1, psnr))
            
            mse, mae, ssim2, psnr = metric(preds2, trues2, True)
            print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim2, psnr))
            
 
            self.model.train()
            self.model1.train()

            return ssim1,ssim2
        elif self.args.dataname == 'Dendrite' :
            self.model.eval()
            self.model1.eval()
            self.model2.eval()
            self.model3.eval()
            vali_pbar = tqdm(vali_loader)
            preds_lst1, trues_lst1 =[], []
            preds_lst2, trues_lst2 =[], []
            preds_lst3, trues_lst3 =[], []
            preds_lst4, trues_lst4 =[], []

            
            for i,(batch_0_1, batch_1_2,batch_2_3,batch_3_4,batch_4_5) in enumerate(vali_pbar):
                pred,true=[],[]
                batch_0_1, batch_1_2,batch_2_3,batch_3_4,batch_4_5 = batch_0_1.to(self.device, dtype=torch.float),batch_1_2.to(self.device, dtype=torch.float),batch_2_3.to(self.device, dtype=torch.float),batch_3_4.to(self.device, dtype=torch.float),batch_4_5.to(self.device, dtype=torch.float)
                pred_1_2 = self.model(batch_0_1)            
                pred_2_3 = self.model1(batch_1_2)                    
                pred_3_4 = self.model2(batch_2_3)                   
                pred_4_5 = self.model3(batch_3_4)
                
                
                preds_lst1.append(pred_1_2.detach().cpu().numpy())
                trues_lst1.append(batch_1_2.detach().cpu().numpy())
                preds_lst2.append(pred_2_3.detach().cpu().numpy())
                trues_lst2.append(batch_2_3.detach().cpu().numpy())
                preds_lst3.append(pred_3_4.detach().cpu().numpy())
                trues_lst3.append(batch_3_4.detach().cpu().numpy())
                preds_lst4.append(pred_4_5.detach().cpu().numpy())
                trues_lst4.append(batch_4_5.detach().cpu().numpy())

            preds1 = np.concatenate(preds_lst1, axis=0)
            trues1 = np.concatenate(trues_lst1, axis=0)
            preds2 = np.concatenate(preds_lst2, axis=0)
            trues2 = np.concatenate(trues_lst2, axis=0)
            preds3 = np.concatenate(preds_lst3, axis=0)
            trues3 = np.concatenate(trues_lst3, axis=0)
            preds4 = np.concatenate(preds_lst4, axis=0)
            trues4 = np.concatenate(trues_lst4, axis=0)
            mse, mae, ssim1, psnr = metric(preds1, trues1, True)
            print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim1, psnr))
            
            mse, mae, ssim2, psnr = metric(preds2, trues2, True)
            print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim2, psnr))
            
            mse, mae, ssim3, psnr = metric(preds3, trues3, True)
            print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim3, psnr))
            
            mse, mae, ssim4, psnr = metric(preds4, trues4, True)
            print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim4, psnr))
            

            self.model.train()
            self.model1.train()
            self.model2.train()
            self.model3.train()

            return ssim1,ssim2,ssim3,ssim4
            
                
                
                
            
        elif self.args.dataname == 'Grain_growth' or self.args.dataname =='Spinodal_decomposition' or self.args.dataname =='Dendrite_growth' or self.args.dataname =='Plane_wave_propagation':
            self.model.eval()
            preds_lst, trues_lst, total_loss = [], [], []

            vali_pbar = tqdm(vali_loader)

            for i, (batch_0_2, batch_2_4) in enumerate(vali_pbar):
                pred, true = [], []

                batch_0_2, batch_2_4 = batch_0_2.to(self.device, dtype=torch.float), batch_2_4.to(
                    self.device, dtype=torch.float)

                pred_2_4 = self.model(batch_0_2)

                loss = self.criterion(pred_2_4, batch_2_4)

                pred.append(pred_2_4.detach().cpu().numpy())

                true.append(batch_2_4.detach().cpu().numpy())

                pred1 = np.concatenate(pred, axis=1)
                true1 = np.concatenate(true, axis=1)
                preds_lst.append(pred1)
                trues_lst.append(true1)

                total_loss.append(loss.mean().item())

            total_loss = np.average(total_loss)
            preds = np.concatenate(preds_lst, axis=0)
            trues = np.concatenate(trues_lst, axis=0)
            

            mse, mae, rmse,ssim, psnr = metric(preds, trues, True)
            print_log('vali mse:{:.8f}, mae:{:.8f}, rmse: {:.8f},ssim:{:.8f}, psnr:{:.8f}'.format(mse, mae, rmse,ssim, psnr))
            self.model.train()
            return ssim
        elif self.args.dataname == 'mmnist':
            for i, (batch_x, batch_y) in enumerate(vali_pbar):
                if i * batch_x.shape[0] > 1000:
                    break

                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x)
                list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                    pred_y, batch_y], [preds_lst, trues_lst]))

                loss = self.criterion(pred_y, batch_y)
                vali_pbar.set_description(
                    'vali loss: {:.4f}'.format(loss.mean().item()))
                total_loss.append(loss.mean().item())

            total_loss = np.average(total_loss)
            preds = np.concatenate(preds_lst, axis=0)
            trues = np.concatenate(trues_lst, axis=0)
            mse, mae, ssim, psnr = metric(preds, trues, True)
            print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
            self.model.train()
            return total_loss
    def test1(self):
        with torch.no_grad():
            if self.args.dataname == 'Grain_growth' or self.args.dataname =='Spinodal_decomposition' or self.args.dataname =='Dendrite_growth' or self.args.dataname =='Plane_wave_propagation':
                self.test(self.vali_loader)
            
            else:
                self.test(self.test_loader)
    def test(self,test_pbar):
       
        if   self.args.dataname == 'p':
            best_model_path = self.path + '/' + 'checkpoint' + '' + '.pth'
            best_model_path1 = self.path + '/' + 'checkpoint' + '1' + '.pth'

            self.model2.load_state_dict(torch.load(best_model_path))
            self.model3.load_state_dict(torch.load(best_model_path1))

            test_pbar = tqdm(test_pbar)
        
            self.model2.eval()
            self.model3.eval()
            preds_lst, trues_lst, total_loss = [], [], []

            for i, (batch_2_4, batch_4_6,batch_6_8) in enumerate(test_pbar):
                pred, true = [], []

                batch_2_4, batch_4_6,batch_6_8= batch_2_4.to(
                    self.device, dtype=torch.float), batch_4_6.to(self.device, dtype=torch.float),batch_6_8.to(self.device, dtype=torch.float)


                pred_4_6 = self.model2(batch_2_4)
                pred_4_6 = pred_4_6.detach()

                pred_6_8 = self.model3(pred_4_6)
                pred_6_8 = pred_6_8.detach()
               
                pred.append(pred_4_6.detach().cpu().numpy())
                pred.append(pred_6_8.detach().cpu().numpy())
                
                true.append(batch_4_6.detach().cpu().numpy())
                true.append(batch_6_8.detach().cpu().numpy())

                pred1 = np.concatenate(pred, axis=1)
                true1 = np.concatenate(true, axis=1)
                preds_lst.append(pred1)
                trues_lst.append(true1)

            preds = np.concatenate(preds_lst, axis=0)
            trues = np.concatenate(trues_lst, axis=0)

            mse, mae, rmse,ssim, psnr = metric(preds, trues, True)
            print_log('vali mse:{:.8f}, mae:{:.8f}, rmse: {:.8f},ssim:{:.8f}, psnr:{:.8f}'.format(mse, mae, rmse,ssim, psnr))

            np.save(self.path + '/' + self.args.model + '_' + self.args.dataname + '1.npy', preds)
            self.model.train()

        elif self.args.dataname == 'Dendrite' :
            best_model_path = self.path + '/' + 'checkpoint' + '' + '.pth'
            best_model_path1 = self.path + '/' + 'checkpoint' + '1' + '.pth'
            best_model_path2 = self.path + '/' + 'checkpoint' + '2' + '.pth'
            best_model_path3 = self.path + '/' + 'checkpoint' + '3' + '.pth'
            
            self.model4.load_state_dict(torch.load(best_model_path))
            self.model5.load_state_dict(torch.load(best_model_path1))
            self.model6.load_state_dict(torch.load(best_model_path2))
            self.model7.load_state_dict(torch.load(best_model_path3))
            
            vali_pbar = tqdm(test_pbar)                
            
            preds_lst, trues_lst, total_loss = [], [], []
            for batch_0_1, batch_1_2,batch_2_3,batch_3_4,batch_4_5 in vali_pbar:
                pred,true=[],[]
                batch_0_1, batch_1_2,batch_2_3,batch_3_4,batch_4_5 = batch_0_1.to(self.device, dtype=torch.float),batch_1_2.to(self.device, dtype=torch.float),batch_2_3.to(self.device, dtype=torch.float),batch_3_4.to(self.device, dtype=torch.float),batch_4_5.to(self.device, dtype=torch.float)
                pred_1_2 = self.model4(batch_0_1)
                pred_1_2 = pred_1_2.detach()
                pred_2_3 = self.model5(pred_1_2) 
                pred_2_3 = pred_2_3.detach()

                pred_3_4 = self.model6(pred_2_3)
                pred_3_4 = pred_3_4.detach()

                pred_4_5 = self.model7(pred_3_4)
                
                pred.append(pred_1_2.detach().cpu().numpy())
                pred.append(pred_2_3.detach().cpu().numpy())
                pred.append(pred_3_4.detach().cpu().numpy())
                pred.append(pred_4_5.detach().cpu().numpy())
                true.append(batch_1_2.detach().cpu().numpy())
                true.append(batch_2_3.detach().cpu().numpy())
                true.append(batch_3_4.detach().cpu().numpy())
                true.append(batch_4_5.detach().cpu().numpy())
                pred1 = np.concatenate(pred, axis=1)
                true1 = np.concatenate(true, axis=1)
                preds_lst.append(pred1)
                trues_lst.append(true1)
            preds = np.concatenate(preds_lst, axis=0)
            trues = np.concatenate(trues_lst, axis=0)
         
            mse, mae, rmse,ssim, psnr = metric(preds, trues, True)
            print_log('vali mse:{:.8f}, mae:{:.8f}, rmse: {:.8f},ssim:{:.8f}, psnr:{:.8f}'.format(mse, mae, rmse,ssim, psnr))
            np.save(self.path + '/' + self.args.model + '_' + self.args.dataname + '.npy', preds)
            
                
                
            
            

            
        elif self.args.dataname == 'Grain_growth' or self.args.dataname =='Spinodal_decomposition'  or self.args.dataname =='Plane_wave_propagation':
            
            best_model_path = self.path + '/' + 'checkpoint' + '' + '.pth'
            self.model1.load_state_dict(torch.load(best_model_path))
            self.model1.eval()
            preds_lst, trues_lst = [], []

            vali_pbar = tqdm(test_pbar)
            

            for batch_input, batch_target in vali_pbar:
                batch_input = batch_input.to(self.device, dtype=torch.float)
                batch_target = batch_target.to(self.device, dtype=torch.float)
                cur_input = batch_input.clone()
                pred_seq = []
               
                with torch.no_grad():
                    for i in range(0, self.args.pre_seq_length, 10):  # pred_step = 10
                        
                        pred_output = self.model1(cur_input.contiguous())  # [B, pred_step, 1, 64, 64]
                        
                        pred_seq.append(pred_output)

                    
                        cur_input = torch.cat([cur_input, pred_output], dim=1)  # [B, T+10, ...]
                        cur_input = cur_input[:, -10:]    
                pred_seq = torch.cat(pred_seq, dim=1)
                preds_lst.append(pred_seq.detach().cpu().numpy())
                trues_lst.append(batch_target.detach().cpu().numpy())

# 
            preds = np.concatenate(preds_lst, axis=0)   # shape: [N, aft_seq_len, 1, 64, 64]
            trues = np.concatenate(trues_lst, axis=0)
            save_dir = '/data/wangyuxin/VMmabaGP/save'
            np.save(os.path.join(save_dir, 'plane_shuffle_pred_90.npy'), preds)
            np.save(os.path.join(save_dir, 'plane_shuffle_truth_90.npy'), trues)
            mse, mae, rmse, ssim, psnr = metric(preds, trues, True)
            print_log('vali1 mse:{:.7f}, mae:{:.7f},rmse:{:.7f}, ssim:{:.7f}, psnr:{:.7f}'.format(mse, mae, rmse, ssim, psnr))


        elif self.args.dataname == 'Dendrite_growth' :
            
            best_model_path = self.path + '/' + 'checkpoint' + '' + '.pth'
            self.model1.load_state_dict(torch.load(best_model_path))
            self.model1.eval()
            preds_lst, trues_lst = [], []

            vali_pbar = tqdm(test_pbar)
            

            for batch_input, batch_target in vali_pbar:
                batch_input = batch_input.to(self.device, dtype=torch.float)
                batch_target = batch_target.to(self.device, dtype=torch.float)
                cur_input = batch_input.clone()
                pred_seq = []
                
                with torch.no_grad():
                    for i in range(0, self.args.pre_seq_length, 10):  # pred_step = 10
                        
                        pred_output = self.model1(cur_input.contiguous())  # [B, pred_step, 1, 64, 64]
                        
                        pred_seq.append(pred_output)

                    
                        cur_input = torch.cat([cur_input, pred_output], dim=1)  # [B, T+10, ...]
                        cur_input = cur_input[:, -10:]    
                pred_seq = torch.cat(pred_seq, dim=1)
                preds_lst.append(pred_seq.detach().cpu().numpy())
                trues_lst.append(batch_target.detach().cpu().numpy())

           
            
            preds = np.concatenate(preds_lst, axis=0)   # shape: [N, aft_seq_len, 1, 64, 64]
            trues = np.concatenate(trues_lst, axis=0)
            save_dir = '/data/wangyuxin/VMmabaGP/save'
            np.save(os.path.join(save_dir, 'spindol_shuffle_pred_90.npy'), preds)
            np.save(os.path.join(save_dir, 'spindol_shuffle_truth_90.npy'), trues)
            mse, mae, rmse, ssim, psnr = metric(preds, trues, True)
            print_log('vali1 mse:{:.7f}, mae:{:.7f},rmse:{:.7f}, ssim:{:.7f}, psnr:{:.7f}'.format(mse, mae, rmse, ssim, psnr))

        elif self.args.dataname == 'mmnist' :
            vali_pbar = tqdm(test_pbar)
            for i, (batch_x, batch_y) in enumerate(vali_pbar):
                if i * batch_x.shape[0] > 1000:
                    break

                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x)
                list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                    pred_y, batch_y], [preds_lst, trues_lst]))

                loss = self.criterion(pred_y, batch_y)
                vali_pbar.set_description(
                    'vali loss: {:.4f}'.format(loss.mean().item()))

            preds = np.concatenate(preds_lst, axis=0)
            trues = np.concatenate(trues_lst, axis=0)
            mse, mae, ssim, psnr = metric(preds, trues, True)
            print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
            np.save(self.path + '/' + self.args.model + '_' + self.args.dataname + '.npy', preds)