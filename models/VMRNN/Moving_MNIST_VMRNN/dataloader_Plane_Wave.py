import torch
import os
from torch.utils.data import Dataset
import numpy as np

from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader

import pdb
class SlidingWindowDataset(Dataset):
   

    def __init__(self, data, pre_seq_len=10, aft_seq_len=10, stride=10,data_name='Dendrite_growth'):
        super().__init__()
        self.pre_seq_len = pre_seq_len
        self.aft_seq_len = aft_seq_len
        self.stride = stride
        self.window_size = pre_seq_len + aft_seq_len
        self.mean = 0
        self.std = 1
        self.data_name = data_name

        self.samples = []  #  (sample_idx, start_idx)

        for sample_idx in range(data.shape[0]):
            for start in range(0, data.shape[1] - self.window_size + 1, stride):
                self.samples.append((sample_idx, start))

        self.data = data  # [N, T, H, W]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_idx, start_idx = self.samples[index]
        clip = self.data[sample_idx, start_idx:start_idx + self.window_size]  # (T, H, W)
        input_seq = clip[:self.pre_seq_len]
        output_seq = clip[self.pre_seq_len:]
        return torch.tensor(input_seq).unsqueeze(1), torch.tensor(output_seq).unsqueeze(1)
        #  [T, 1, H, W]



class LongSequenceDataset(Dataset):
    def __init__(self, data, pre_seq_len=10, aft_seq_len=10, stride=10):
        super().__init__()
        self.data = data  # shape: (T, 1, H, W)
        self.pre_seq_len = pre_seq_len
        self.aft_seq_len = aft_seq_len
        self.window_size = pre_seq_len + aft_seq_len

        total_frames = data.shape[0]
        self.samples = []

        
        for start in range(0, total_frames - self.window_size + 1, self.window_size):
            self.samples.append(start)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        start_idx = self.samples[index]
        clip = self.data[start_idx:start_idx + self.window_size]  # shape: [T, 1, H, W]
        input_seq = clip[:self.pre_seq_len]
        output_seq = clip[self.pre_seq_len:]
        return torch.tensor(input_seq),torch.tensor(output_seq)



def load_data(batch_size, val_batch_size, data_root, 
              stride=10, num_workers=4, pre_seq_length=None, aft_seq_length=None,in_shape=None,distributed=False,use_augment=False,
              use_prefetcher=False, drop_last=False):

    #train = np.load(os.path.join(data_root, 'Grain_growth/train.npz'))['input_raw_data']  # [N, T, H, W]
    #val = np.load(os.path.join(data_root, 'Grain_growth/valid.npz'))['input_raw_data']
    
    train = np.load(os.path.join(data_root, 'Plane_wave_propagation/train.npz'))['data'][:,:,:,:,0]
    val = np.load(os.path.join(data_root, 'Plane_wave_propagation/valid.npz'))['data'][:,:,:,:,0]
    test = np.load(os.path.join(data_root, 'Plane_wave_propagation/test.npz'))['data']

    #train_set = LongSequenceDataset(train, pre_seq_length, aft_seq_length)
    #val_set = LongSequenceDataset(val, pre_seq_length, aft_seq_length)
    
    train_set = SlidingWindowDataset(train, pre_seq_length, aft_seq_length, stride)
    val_set = SlidingWindowDataset(val, pre_seq_length, aft_seq_length, stride)
    test_set = SlidingWindowDataset(test, pre_seq_length, aft_seq_length, stride)

    dataloader_train = create_loader(train_set, batch_size=batch_size, shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True, num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)

    dataloader_val = create_loader(val_set, batch_size=val_batch_size, shuffle=False, is_training=False,
                                   pin_memory=True, drop_last=drop_last, num_workers=num_workers,
                                   distributed=distributed, use_prefetcher=use_prefetcher)

    dataloader_test = create_loader(test_set, batch_size=val_batch_size, shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last, num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_val, dataloader_test


if __name__ == '__main__':
    dataloader_train, dataloader_valid, dataloader_test = load_data(
        batch_size=8,
        val_batch_size=4,
        data_root='../../data/',
        pre_seq_length=10,
        aft_seq_length=10,
        stride=10
    )
    
    total_batches = 0
    for x, y in dataloader_train:
        total_batches += 1
    print(total_batches)
        
