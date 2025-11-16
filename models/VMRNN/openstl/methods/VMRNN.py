import time
import torch
import torch.nn as nn
from openstl.models import VMRNN_D_Model, VMRNN_B_Model
from openstl.utils import (reduce_tensor, reshape_patch, reshape_patch_back,
                           reserve_schedule_sampling_exp, schedule_sampling)
from timm.utils import AverageMeter
from tqdm import tqdm
from .base_method import Base_method
import pdb
class VMRNN_D(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        super(VMRNN_D, self).__init__(args, device, steps_per_epoch)
        # depths_downsample = [int(x) for x in args.depths_downsample.split(',')]
        # depths_upsample = [int(x) for x in args.depths_upsample.split(',')]
        # num_heads = [int(x) for x in args.num_heads.split(',')]
        self.model = self._build_model(args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        depths_downsample = [int(x) for x in args.depths_downsample.split(',')]
        depths_upsample = [int(x) for x in args.depths_upsample.split(',')]
        num_heads = [int(x) for x in args.num_heads.split(',')]
        # Ensure that VMRNN_D_Model is correctly initialized with the expected parameters
        return VMRNN_D_Model(depths_downsample, depths_upsample, num_heads, args).to(self.device)
    
    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model"""
        
        
        ### 迭代进行预测
        target_seq_length = batch_y.shape[1]
        step_length = batch_x.shape[1]

        
        if target_seq_length % step_length != 0:
            raise ValueError("目标序列长度必须是单次预测步长的整数倍")
            
        num_iterations = target_seq_length // step_length
         # 用于存储每一次迭代的预测结果
        all_predictions = []
        
        # 将初始输入作为第一次迭代的输入
        current_input = batch_x
        
        #for i in range(num_iterations):
            # pdb.set_trace()
            # current_output = batch_y[:,i*10:i*10+10,:,:,:]
            #test_ims = torch.cat([current_input, current_output], dim=1).permute(0, 1, 3, 4, 2).contiguous()
            #img_gen, _ = self.model(test_ims, return_loss=False)
            #pred_y = img_gen[:, -step_length:].permute(0, 1, 4, 2, 3).contiguous()
            #all_predictions.append(pred_y) 
            #current_input  = pred_y
        
        #all_predictions = torch.cat(all_predictions,dim =1 )
        
        """Forward the model"""
        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        img_gen, _ = self.model(test_ims, return_loss=False)
        pred_y = img_gen[:, -self.args.aft_seq_length:].permute(0, 1, 4, 2, 3).contiguous()
        
        return pred_y     
        
    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')

            ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()


            with self.amp_autocast():
                img_gen, loss = self.model(ims)

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta


class VMRNN_B(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        super(VMRNN_B, self).__init__(args, device, steps_per_epoch)
        self.model = self._build_model(args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        return VMRNN_B_Model(args).to(self.device)
    
    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model"""
        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        img_gen, _ = self.model(test_ims, return_loss=False)
        pred_y = img_gen[:, -self.args.aft_seq_length:].permute(0, 1, 4, 2, 3).contiguous()
        return pred_y
     
        
    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')

            ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()


            with self.amp_autocast():
                img_gen, loss = self.model(ims)

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta