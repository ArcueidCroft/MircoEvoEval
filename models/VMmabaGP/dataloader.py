# -*- coding: utf-8 -*-
import numpy as np
from torch.utils.data import DataLoader
import pdb

class Original_dataLoader(object):
    def __init__(self,mode='train',data_dir='../../data/Spinodal_decomposition',pre_seq_length=10,aft_seq_length=10,stride = 4):
        train = np.load(data_dir+'/train.npy')
        train = train[:, :, np.newaxis, :, :]
        self.train_1 = train[:, :pre_seq_length]
        self.train_2 = train[:, pre_seq_length:pre_seq_length * 2]
       




        test = np.load(data_dir+'/valid.npy')
        test = test[:, :, np.newaxis, :, :]

        self.test_1 = test[:, :pre_seq_length]
        self.test_2 = test[:, pre_seq_length:pre_seq_length * 2]
      

        val = np.load(data_dir+'/test.npy')
        
        
        val = val[:, :, np.newaxis, :, :]
        # pdb.set_trace()
        self.val_1 = val[:, :pre_seq_length]
        self.val_2 = val[:, pre_seq_length:pre_seq_length+aft_seq_length]

        self.mode = mode

    def __len__(self):

        if self.mode == "train":
            return self.train_1.shape[0]
        elif self.mode == 'val':
            return self.val_inputs.shape[0]
        elif self.mode == 'test':
            return self.test_1.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
          return self.train_1[index],self.train_2[index]
        elif self.mode == "test":
          return self.test_1[index],self.test_2[index]
        elif self.mode == "val":
          return self.val_inputs[index], self.val_targets[index]



class Grain_growth_dataLoader(object):
    def __init__(self,mode='train',data_dir='../../data/Grain_growth',pre_seq_length=10,aft_seq_length=10,stride = 4):
        train = np.load(data_dir+'/train.npy')
        train = train[:, :, np.newaxis, :, :]
        self.train_1 = train[:, :pre_seq_length]
        self.train_2 = train[:, pre_seq_length:pre_seq_length * 2]
       




        test = np.load(data_dir+'/test.npy')
        test = test[:, :, np.newaxis, :, :]

        self.test_1 = test[:, :pre_seq_length]
        self.test_2 = test[:, pre_seq_length:pre_seq_length * 2]
      

        
        val = np.load(data_dir+'/val.npz')['data']
        
        val = val[:, :, np.newaxis, :, :]
        self.val_inputs = []
        self.val_targets = []

        total_frames = val.shape[1]
        clip_len = pre_seq_length + aft_seq_length
        

        N = val.shape[0]
        for sample_idx in range(N):  # N = 100
            for start in range(0, total_frames - clip_len + 1, stride):
                input_seq = val[sample_idx:sample_idx+1, start : start + aft_seq_length]
                target_seq = val[sample_idx:sample_idx+1, start + aft_seq_length: start + clip_len]
                self.val_inputs.append(input_seq)
                self.val_targets.append(target_seq)
                


        self.val_inputs = np.concatenate(self.val_inputs, axis=0)
        self.val_targets = np.concatenate(self.val_targets, axis=0)
     


        self.mode = mode

    def __len__(self):

        if self.mode == "train":
            return self.train_1.shape[0]
        elif self.mode == 'val':
            return self.val_inputs.shape[0]
        elif self.mode == 'test':
            return self.test_1.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
          return self.train_1[index],self.train_2[index]
        elif self.mode == "test":
          return self.test_1[index],self.test_2[index]
        elif self.mode == "val":
          return self.val_inputs[index], self.val_targets[index]


class Spinodal_decomposition_dataLoader(object):
    def __init__(self,mode='train',data_dir='../../data/Spinodal_decomposition',pre_seq_length=10,aft_seq_length=10,stride = 4):
        train = np.load(data_dir+'/train.npy')
        train = train[:, :, np.newaxis, :, :]
        self.train_1 = train[:, :pre_seq_length]
        self.train_2 = train[:, pre_seq_length:pre_seq_length * 2]
       




        test = np.load(data_dir+'/test.npy')
        test = test[:, :, np.newaxis, :, :]

        self.test_1 = test[:, :pre_seq_length]
        self.test_2 = test[:, pre_seq_length:pre_seq_length * 2]
      

        val = np.load(data_dir+'/val.npz')['data']
        
        
        val = val[:, :, np.newaxis, :, :]
        self.val_inputs = []
        self.val_targets = []

        

        total_frames = val.shape[1]
        clip_len = pre_seq_length + aft_seq_length
        
        N = val.shape[0]
        for sample_idx in range(N):  # N = 100
            for start in range(0, total_frames - clip_len + 1, stride):
                input_seq = val[sample_idx:sample_idx+1, start : start + aft_seq_length]
                target_seq = val[sample_idx:sample_idx+1, start + aft_seq_length: start + clip_len]
                self.val_inputs.append(input_seq)
                self.val_targets.append(target_seq)
                


        self.val_inputs = np.concatenate(self.val_inputs, axis=0)
        self.val_targets = np.concatenate(self.val_targets, axis=0)
        
        # pdb.set_trace()


        self.mode = mode

    def __len__(self):

        if self.mode == "train":
            return self.train_1.shape[0]
        elif self.mode == 'val':
            return self.val_inputs.shape[0]
        elif self.mode == 'test':
            return self.test_1.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
          return self.train_1[index],self.train_2[index]
        elif self.mode == "test":
          return self.test_1[index],self.test_2[index]
        elif self.mode == "val":
          return self.val_inputs[index], self.val_targets[index]

    
    
class Dendrite_growth_dataLoader(object):
    def __init__(self,mode='train',data_dir='../../data/Dendrite_growth',pre_seq_length=10,aft_seq_length=10, stride = 4):
        train = np.load(data_dir+'/train.npy')
        train = train[:, :, np.newaxis, :, :]
        self.train_1 = train[:, :pre_seq_length]
        self.train_2 = train[:, pre_seq_length:pre_seq_length * 2]
       




        test = np.load(data_dir+'/test.npy')
        test = test[:, :, np.newaxis, :, :]

        self.test_1 = test[:, :pre_seq_length]
        self.test_2 = test[:, pre_seq_length:pre_seq_length * 2]
      

        
        val = np.load(data_dir+'/val.npz')['data']
        
        val = val[:, :, np.newaxis, :, :]
        self.val_inputs = []
        self.val_targets = []

        total_frames = val.shape[1]
        clip_len = pre_seq_length + aft_seq_length
        

        N = val.shape[0]
        for sample_idx in range(N):  # N = 100
            for start in range(0, total_frames - clip_len + 1, stride):
                input_seq = val[sample_idx:sample_idx+1, start : start + aft_seq_length]
                target_seq = val[sample_idx:sample_idx+1, start + aft_seq_length: start + clip_len]
                self.val_inputs.append(input_seq)
                self.val_targets.append(target_seq)
                


        self.val_inputs = np.concatenate(self.val_inputs, axis=0)
        self.val_targets = np.concatenate(self.val_targets, axis=0)
        
     


        self.mode = mode

    def __len__(self):

        if self.mode == "train":
            return self.train_1.shape[0]
        elif self.mode == 'val':
            return self.val_inputs.shape[0]
        elif self.mode == 'test':
            return self.test_1.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
          return self.train_1[index],self.train_2[index]
        elif self.mode == "test":
          return self.test_1[index],self.test_2[index]
        elif self.mode == "val":
          return self.val_inputs[index], self.val_targets[index]




class p_dataLoader(object):
    def __init__(self,mode='train',data_dir='../../data/data_p',pre_seq_length=10):
        train_array = np.load(data_dir+'/train_array.npy')
        train_array = train_array[:, :, np.newaxis, :, :]
        
        test_array = np.load(data_dir+'/test_array.npy')
        test_array = test_array[:, :, np.newaxis, :, :]
        
        
        validation_array = np.load(data_dir+'/validation_array.npy')
        validation_array = validation_array[:, :, np.newaxis, :, :]

        
        self.train_1 = train_array[:,pre_seq_length:pre_seq_length*2:2]
        self.train_2 = train_array[:, pre_seq_length*2:pre_seq_length*3:2]
        self.train_3 = train_array[:, pre_seq_length*3:pre_seq_length*4:2]



        self.val_1 = validation_array[:, pre_seq_length:pre_seq_length*2:2]
        self.val_2 = validation_array[:, pre_seq_length*2:pre_seq_length*3:2]
        self.val_3 = validation_array[:, pre_seq_length*3:pre_seq_length*4:2]


        self.test_1 = test_array[:, pre_seq_length:pre_seq_length*2:2]
        self.test_2 = test_array[:, pre_seq_length*2:pre_seq_length*3:2]
        self.test_3 = test_array[:, pre_seq_length*3:pre_seq_length*4:2]
        
        self.mode = mode

    def __len__(self):

        if self.mode == "train":
            return self.train_1.shape[0]
           
        elif self.mode == 'val':
            return self.val_1.shape[0]
        elif self.mode == 'test':
            return self.test_1.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
          return self.train_1[index],self.train_2[index],self.train_3[index]
        elif self.mode == "test":
          return self.test_1[index],self.test_2[index],self.test_3[index]
        elif self.mode == "val":
          return self.val_1[index],self.val_2[index],self.val_3[index]

    
    
    

    
    
    



def get_loader_segment( batch_size, pre_seq_length=10, aft_seq_length=10, mode='train', dataset='KDD'):
    if (dataset == 'Dendrite_growth'):
        dataset = Original_dataLoader(mode=mode,data_dir='../../data/Dendrite_growth',pre_seq_length=pre_seq_length,aft_seq_length = aft_seq_length)
    elif (dataset == 'Grain_growth'):
        dataset = Original_dataLoader(mode=mode,data_dir='../../data/Grain_growth',pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)
    elif (dataset == 'Dendrite'):
        dataset = Original_dataLoader(mode=mode,data_dir='data/Dendrite',pre_seq_length=pre_seq_length,aft_seq_length=aft_seq_length)
    elif (dataset =='p'): 
        dataset = p_dataLoader(mode=mode,data_dir='data/data_p',pre_seq_length=pre_seq_length)
    elif (dataset =='Spinodal_decomposition'): 
        dataset = Original_dataLoader(mode=mode,data_dir='../../data/Spinodal_decomposition',pre_seq_length=pre_seq_length,aft_seq_length=aft_seq_length)
    elif (dataset =='Plane_wave_propagation'): 
        dataset = Original_dataLoader(mode=mode,data_dir='../../data/Plane_wave_propagation',pre_seq_length=pre_seq_length,aft_seq_length=aft_seq_length)
        
        


    shuffle = False
    if mode == 'train':
        shuffle = False
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=8)
                         
    return data_loader
























