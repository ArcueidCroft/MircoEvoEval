import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch

from torch.utils.data import Dataset





class Dataloader(Dataset):
    def __init__(self, data_path, n_frames_input=10, n_frames_output=10, step= 4 ,transform=None):
        
        data = np.load(data_path)
        data_name = data.files

        self.data = data[data_name[0]]   # [N, T, C, W]
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.transform = transform
        self.total_frames = n_frames_input + n_frames_output
        self.step = step

        self.samples = []
        
        for sample_idx in range(len(self.data)):
            T = self.data[sample_idx].shape[0]
            for start in range(0, T - self.total_frames + 1, step):
                self.samples.append((sample_idx, start))

        
        assert self.data.shape[1] >= (n_frames_input + n_frames_output), \
            f"Each sequence requires at least {n_frames_input + n_frames_output} frames, but only {sequence_len} were provided"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_idx, start = self.samples[index]
        sequence = self.data[sample_idx]  # [T, H, W]

        clip = sequence[start:start + self.total_frames]
        input_frames = clip[:self.n_frames_input]
        output_frames = clip[self.n_frames_input:]

        input_tensor = torch.from_numpy(input_frames ).unsqueeze(1).float()  # [T_in, 1, H, W]
        output_tensor = torch.from_numpy(output_frames ).unsqueeze(1).float()  # [T_out, 1, H, W]

        if self.transform:
            input_tensor = self.transform(input_tensor)
            output_tensor = self.transform(output_tensor)

        return [index, output_tensor, input_tensor]


class Plane(Dataset):
    def __init__(self, data_path, n_frames_input=10, n_frames_output=10, step=4, transform=None):
        """
        Custom dataset class for loading time-series image data with shape [N, T, H, W].

        :param data_path: Path to the .npy file
        :param n_frames_input: Number of input frames
        :param n_frames_output: Number of output (predicted) frames
        :param transform: Optional transform or data augmentation
        """
        data = np.load(data_path)
        data_name = data.files

        self.data = data[data_name[0]][:, :, :, :, 0]  ##   only for  Plane_wave_propagation
        # self.data = data[data_name[0]]  for spindol and xuehua [N, T, C, W]
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.transform = transform
        self.total_frames = n_frames_input + n_frames_output
        self.step = step

        self.samples = []
        for sample_idx in range(len(self.data)):
            T = self.data[sample_idx].shape[0]
            for start in range(0, T - self.total_frames + 1, step):
                self.samples.append((sample_idx, start))

        
        assert self.data.shape[1] >= (n_frames_input + n_frames_output), \
            f"Each sequence requires at least {n_frames_input + n_frames_output} frames, but only {sequence_len} were provided"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_idx, start = self.samples[index]
        sequence = self.data[sample_idx]  # [T, H, W]

        clip = sequence[start:start + self.total_frames]
        input_frames = clip[:self.n_frames_input]
        output_frames = clip[self.n_frames_input:]

        input_tensor = torch.from_numpy(input_frames).unsqueeze(1).float()  # [T_in, 1, H, W]
        output_tensor = torch.from_numpy(output_frames).unsqueeze(1).float()  # [T_out, 1, H, W]

        if self.transform:
            input_tensor = self.transform(input_tensor)
            output_tensor = self.transform(output_tensor)

        return [index, output_tensor, input_tensor]



class Grain(Dataset):
    def __init__(self, data_path, n_frames_input=10, n_frames_output=10, sequence_len=20, transform=None):
        """
        Custom dataset class for loading data shaped as [N*T, C, H, W], where each sequence of sequence_len frames
        forms one complete sample.
        
        :param data_path: Path to the .npy file
        :param n_frames_input: Number of input frames
        :param n_frames_output: Number of output (predicted) frames
        :param sequence_len: Total number of frames in each full sequence (e.g., 20)
        :param transform: Optional transform or data augmentation
        """
        data = np.load(data_path)  # [30400, 1, 64, 64]

        data_name = data.files

        raw_data = data[data_name[-1]]
        assert raw_data.shape[0] % sequence_len == 0, "Total number of frames must be divisible by sequence_len"

        self.n_sequences = raw_data.shape[0] // sequence_len
        self.data = raw_data.reshape(self.n_sequences, sequence_len, *raw_data.shape[1:])  # [N, 20, 1, 64, 64]

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.transform = transform

        assert sequence_len >= (n_frames_input + n_frames_output), \
            f"Each sequence requires at least {n_frames_input + n_frames_output} frames, but only {sequence_len} were provided"


    def __len__(self):
        return self.n_sequences

    def __getitem__(self, index):
        clip = self.data[index]  # [20, 1, 64, 64]
        input_frames = clip[:self.n_frames_input]      # [10, 1, 64, 64]
        output_frames = clip[self.n_frames_input:self.n_frames_input + self.n_frames_output]  # [10, 1, 64, 64]

        input_tensor = torch.from_numpy(input_frames).float()
        output_tensor = torch.from_numpy(output_frames).float()

        if self.transform:
            input_tensor = self.transform(input_tensor)
            output_tensor = self.transform(output_tensor)

        return [index, output_tensor, input_tensor]

