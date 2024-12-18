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
        # time1 = time.time()
        item_addr = self.data_addr[item]
        item_label = self.direction_label[item]
        # print("item:" + str(item_addr))
        # print("load time1:" + str(time.time() - time1))
        # time1 = time.time()

        # spectrum shape is mic_num, nfft / 2, frames
        # addr = os.path.join(item_addr, 'complex_spectrum.npy')
        addr = os.path.join(item_addr, 'signals.npy')
        time_series = np.load(addr, allow_pickle=True)
        # _, time_series = self.wgn(time_series, time_series[0], 0)
        # time_series = self.make_noise(time_series, 0)
        spectrum = fft(time_series)
        spectrum = spectrum[..., range(spectrum.shape[-1] // 2 + 1)]
        real = np.real(spectrum).astype(np.float32)
        imag = np.imag(spectrum).astype(np.float32)
        real_imag = np.concatenate((real, imag), axis=1)
        real_imag = np.transpose(real_imag, (1, 0))
        real_imag = np.reshape(real_imag, (real_imag.shape[0], 10, 10))

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

    @staticmethod
    def make_noise(x, snr):
        for i in range(x.shape[0]):
            Ps = np.mean(np.power(x[i], 2))
            Pn = Ps / (np.power(10, snr / 10))

            # input values
            beta = 2  # the exponent: 0=white noite; 1=pink noise;  2=red noise (also "brownian noise")
            samples = 512  # number of samples to generate (time series extension)
            noise = cn.powerlaw_psd_gaussian(beta, samples)
            noise = noise.astype(np.float32)

            # plt.subplot(221)
            # plt.plot(x[i])

            fs = 16000
            b, a = signal.butter(4, 800 / fs)
            filter_noise = signal.filtfilt(b, a, noise)
            cur_Pn = np.mean(np.power(filter_noise, 2))
            filter_noise = filter_noise * np.sqrt(Pn / cur_Pn)

            # plt.subplot(222)
            # plt.plot(filter_noise)

            x[i] += filter_noise

            # plt.subplot(223)
            # plt.plot(x[i])
            # plt.show()

        return x.astype(np.float32)

    @staticmethod
    def wgn(x, label, snr):
        Ps = np.mean(np.power(label, 2))
        Pn = Ps / (np.power(10, snr / 10))
        noise = np.random.randn(x.shape[0], x.shape[1]) * np.sqrt(Pn)
        return noise, x + noise
