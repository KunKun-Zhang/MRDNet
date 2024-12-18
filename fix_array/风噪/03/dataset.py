import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time
import math
import colorednoise as cn
from scipy import signal
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt

# 子阵交织 经过风噪训练

class WavDataset(Dataset):
    def __init__(self, file_path):
        self.data_addr = []
        self.direction_label = []

        direction = os.listdir(file_path)
        for one_direction in direction:
            file_directory = os.path.join(file_path, one_direction)
            file_list = os.listdir(file_directory)
            for file in file_list:
                join_file = os.path.join(file_directory, file)
                self.data_addr.append(join_file)
                self.direction_label.append(one_direction)

    def __getitem__(self, item):
        item_addr = self.data_addr[item]
        item_label = self.direction_label[item]
        addr = os.path.join(item_addr, 'signals.npy')
        time_series = np.load(addr, allow_pickle=True).item()['noise_signal']
        time_series = np.clip(time_series, -1.0, 1.0)

        spectrum = fft(time_series)
        spectrum = spectrum[..., range(spectrum.shape[-1] // 2 + 1)]
        real = np.real(spectrum).astype(np.float32)
        imag = np.imag(spectrum).astype(np.float32)
        real_imag = np.concatenate((real[:, None, ...], imag[:, None, ...]), axis=1)
        real_imag = real_imag[:, None, ...]
        # real_imag = np.transpose(real_imag, (1, 0))
        # real_imag = np.reshape(real_imag, (real_imag.shape[0], 10, 10))

        item_label = int(item_label)
        # label_torch = torch.zeros(size=(360,))
        # label_torch[item_label] = 1
        label_torch = self.label_smooth(item_label)

        return real_imag, label_torch, item_label
        # return read_and_image_spectrum, label_torch, item_label

    def __len__(self):
        return len(self.data_addr)

    @staticmethod
    def label_smooth(ground_truth):
        label = np.zeros(360)
        curve = 2
        for i in range(label.shape[0]):
            if abs(i - ground_truth) <= curve:
                label[i] = max(label[i], np.exp(-1 * (i - ground_truth) ** 2 / curve ** 2))
            elif abs(i + 360 - ground_truth) <= curve:
                label[i] = max(label[i], np.exp(-1 * (i + 360 - ground_truth) ** 2 / curve ** 2))
            elif abs(i - 360 - ground_truth) <= curve:
                label[i] = max(label[i], np.exp(-1 * (i - 360 - ground_truth) ** 2 / curve ** 2))

        label = label / np.sum(label)
        return label.astype(np.float32)
