import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
#from numpy.lib.polynomial import roots
#   import torch.utils.data as data
from torch.utils.data import Dataset
import pdb

class OriginalDataloader(Dataset):
    def __init__(self, data_path, n_frames_input=10, n_frames_output=10, step= 4 ,transform=None):
        data = np.load(data_path)

        self.data = data   # [N, T, C, W]
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.transform = transform
        self.total_frames = n_frames_input + n_frames_output
        self.step = step

        self.samples = []
        #pdb.set_trace()
        for sample_idx in range(len(self.data)):
            self.samples.append(sample_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_idx = self.samples[index]
        sequence = self.data[sample_idx]  # [T, H, W]

        clip = sequence
        input_frames = clip[:self.n_frames_input]
        output_frames = clip[self.n_frames_input:]

        input_tensor = torch.from_numpy(input_frames ).unsqueeze(1).float()  # [T_in, 1, H, W]
        output_tensor = torch.from_numpy(output_frames ).unsqueeze(1).float()  # [T_out, 1, H, W]

        if self.transform:
            input_tensor = self.transform(input_tensor)
            output_tensor = self.transform(output_tensor)

        return [index, output_tensor, input_tensor]


# # class Dataloader(Dataset):
#     def __init__(self, data_path, n_frames_input=10, n_frames_output=10, step= 4 ,transform=None):
#         data = np.load(data_path)
#         data_name = data.files

#         self.data = data[data_name[0]]   # [N, T, C, W]
#         self.n_frames_input = n_frames_input
#         self.n_frames_output = n_frames_output
#         self.transform = transform
#         self.total_frames = n_frames_input + n_frames_output
#         self.step = step

#         self.samples = []
#         #pdb.set_trace()
#         for sample_idx in range(len(self.data)):
#             T = self.data[sample_idx].shape[0]
#             for start in range(0, T - self.total_frames + 1, step):
#                 self.samples.append((sample_idx, start))

#         assert self.data.shape[1] >= (n_frames_input + n_frames_output), \
#             f"seq need {n_frames_input + n_frames_output} frames, but only have {self.data.shape[1]} frames。"

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, index):
#         sample_idx, start = self.samples[index]
#         sequence = self.data[sample_idx]  # [T, H, W]

#         clip = sequence[start:start + self.total_frames]
#         input_frames = clip[:self.n_frames_input]
#         output_frames = clip[self.n_frames_input:]

#         input_tensor = torch.from_numpy(input_frames ).unsqueeze(1).float()  # [T_in, 1, H, W]
#         output_tensor = torch.from_numpy(output_frames ).unsqueeze(1).float()  # [T_out, 1, H, W]

#         if self.transform:
#             input_tensor = self.transform(input_tensor)
#             output_tensor = self.transform(output_tensor)

#         return [index, output_tensor, input_tensor]
        

# # class Plane(Dataset):
#     def __init__(self, data_path, n_frames_input=10, n_frames_output=10, step=4, transform=None):
#         data = np.load(data_path)
#         data_name = data.files

#         self.data = data[data_name[0]][:, :, :, :, 0]  ##   only for  Plane_wave_propagation
#         # self.data = data[data_name[0]]  for spindol and xuehua [N, T, C, W]
#         self.n_frames_input = n_frames_input
#         self.n_frames_output = n_frames_output
#         self.transform = transform
#         self.total_frames = n_frames_input + n_frames_output
#         self.step = step

#         self.samples = []
#         for sample_idx in range(len(self.data)):
#             T = self.data[sample_idx].shape[0]
#             for start in range(0, T - self.total_frames + 1, step):
#                 self.samples.append((sample_idx, start))

#         # 检查数据长度是否足够
#         assert self.data.shape[1] >= (n_frames_input + n_frames_output), \
#             f"seq need {n_frames_input + n_frames_output} frames, but only have {self.data.shape[1]} frames。"

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, index):
#         sample_idx, start = self.samples[index]
#         sequence = self.data[sample_idx]  # [T, H, W]

#         clip = sequence[start:start + self.total_frames]
#         input_frames = clip[:self.n_frames_input]
#         output_frames = clip[self.n_frames_input:]

#         input_tensor = torch.from_numpy(input_frames).unsqueeze(1).float()  # [T_in, 1, H, W]
#         output_tensor = torch.from_numpy(output_frames).unsqueeze(1).float()  # [T_out, 1, H, W]

#         if self.transform:
#             input_tensor = self.transform(input_tensor)
#             output_tensor = self.transform(output_tensor)

#         return [index, output_tensor, input_tensor]



# # class Grain(Dataset):
#     def __init__(self, data_path, n_frames_input=10, n_frames_output=10, sequence_len=20, transform=None):
#         data = np.load(data_path)  # [30400, 1, 64, 64]

#         data_name = data.files

#         raw_data = data[data_name[-1]]
#         assert raw_data.shape[0] % sequence_len == 0, "r(frames / sequence_len) != 0"

#         self.n_sequences = raw_data.shape[0] // sequence_len
#         self.data = raw_data.reshape(self.n_sequences, sequence_len, *raw_data.shape[1:])  # [N, 20, 1, 64, 64]

#         self.n_frames_input = n_frames_input
#         self.n_frames_output = n_frames_output
#         self.transform = transform

#         assert sequence_len >= (n_frames_input + n_frames_output), \
#             f" {n_frames_input + n_frames_output} frames, with  {sequence_len} frames。"

#     def __len__(self):
#         return self.n_sequences

#     def __getitem__(self, index):
#         clip = self.data[index]  # [20, 1, 64, 64]
#         input_frames = clip[:self.n_frames_input]      # [10, 1, 64, 64]
#         output_frames = clip[self.n_frames_input:self.n_frames_input + self.n_frames_output]  # [10, 1, 64, 64]

#         input_tensor = torch.from_numpy(input_frames).float()
#         output_tensor = torch.from_numpy(output_frames).float()

#         if self.transform:
#             input_tensor = self.transform(input_tensor)
#             output_tensor = self.transform(output_tensor)

#         return [index, output_tensor, input_tensor]
