import numpy as np
import glob
import os
from multiprocessing import Process, Manager, Lock
import math
import pyroomacoustics as pra
import random
import time
from scipy import signal
from scipy.fft import fft, fftfreq
# from matplotlib import pyplot as plt
from enum import Enum

################
# 单语音多次利用 全角度 只保存单帧

# directionEnum = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
directionEnum = [i for i in range(360)]


def open_vad_txt(txt_path):
    with open(txt_path, 'r') as f:
        data = f.read().split('\n')
        source_signal = [float(data[i]) for i in range(0, min(len(data), 16000))]
    # with open(txt_path, 'r') as f:
    #     data = f.read().split('\n')
    #     if len(data) >= 32000:
    #         source_signal = [float(data[i]) for i in range(0, 32000)]
    #     else:
    #         source_signal = np.zeros(shape=(32000,))
    #         for i in range(0, len(data)):
    #             source_signal[i] = float(data[i])
    return source_signal


def framesig(signal, wlen=512, inc=256):
    signal_len = signal.shape
    if signal_len[1] < wlen:
        nf = 1
    else:
        nf = int(np.ceil((1.0 * signal_len[1] - wlen + inc) / inc))
    pad_len = int((nf - 1) * inc + wlen)
    zeros = np.zeros((signal_len[0], pad_len - signal_len[1]))
    pad_signal = np.concatenate((signal, zeros), axis=1)
    indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T  # 每行是单帧的idx
    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[:, indices]
    return frames


def micro_center(room_dim, location_bias):
    safe_location = room_dim * 0.1
    center_x = micro_center_sub_function(location_bias[0], room_dim[0], safe_location[0])
    center_y = micro_center_sub_function(location_bias[1], room_dim[1], safe_location[1])
    center = np.r_[center_x, center_y]
    return center


def micro_center_sub_function(location_bias, edge, safe_location):
    if location_bias >= 0:
        center = random.randint(math.ceil(safe_location), int(edge - location_bias))
    else:
        center = random.randint(math.ceil(-location_bias), int(edge - safe_location))
    return center


def check_file_already_exist(file_name, one_path2):
    file_list = os.listdir(one_path2)
    if file_name in file_list:
        return False
    else:
        return True


def produce_covariance_matrix(spectrum, freq_bins):
    spectrum = np.transpose(spectrum, axes=[2, 1, 0])
    # select frequency bins
    spectrum = spectrum[..., list(freq_bins), :]
    # Compute PSD and average over time frame
    C_hat = np.matmul(spectrum[..., None], np.conjugate(spectrum[..., None, :]))
    # Average over time-frames
    C_hat = np.mean(C_hat, axis=0)
    return C_hat


def create_room_box(direction, file_path, m=12, n=12, c=343.0, fs=16000, nfft=512, room_dim=np.r_[10.0, 8.0],
                    jianju=0.005, snr=0):
    distance = random.randint(3, 5)
    azimuth = direction / 180.0 * np.pi
    # sigma2 = 10 ** (-snr / 10) / (4.0 * np.pi * distance) ** 2
    # aroom = pra.ShoeBox(room_dim, fs=fs, max_order=0, sigma2_awgn=sigma2)
    aroom = pra.ShoeBox(room_dim, fs=fs, max_order=0)

    location_bias = distance * np.r_[np.cos(azimuth), np.sin(azimuth)]
    center = micro_center(room_dim, location_bias)
    source_location = center + location_bias
    source_signal = open_vad_txt(file_path)
    b, a = signal.butter(4, 800 / fs)
    source_signal = signal.filtfilt(b, a, source_signal)
    aroom.add_source(source_location, signal=source_signal)

    microphone_list = pra.square_2D_array(center, m, n, 0, d=jianju)
    aroom.add_microphone_array(pra.MicrophoneArray(microphone_list, fs=aroom.fs))
    aroom.simulate(snr=snr)

    # spectrum = np.array(
    #     [
    #         pra.transform.stft.analysis(signal, nfft, nfft // 2).T
    #         for signal in aroom.mic_array.signals
    #     ]
    # )
    # spectrum = spectrum[:, :, :127]
    # freq_bins = np.arange(0, spectrum.shape[1])

    # doa = pra.doa.MUSIC(microphone_list, fs=fs, nfft=nfft, c=c)
    # doa.locate_sources(spectrum, freq_bins=freq_bins)
    # music_numpy = doa.grid.values
    #
    # covariance_matrix = produce_covariance_matrix(spectrum, freq_bins)
    #
    signals = aroom.mic_array.signals
    signals = framesig(signals)
    music_numpy = 0
    covariance_matrix = 0
    spectrum = 0

    return spectrum, music_numpy, microphone_list, covariance_matrix, signals


def save_complex_spectrum_and_music(save_path, spectrum, music, micro_list, covariance_matrix, signals):
    spectrum_path = os.path.join(save_path, 'complex_spectrum.npy')
    maxFrame_spectrum_path = os.path.join(save_path, 'maxFrame_spectrum.npy')
    music_path = os.path.join(save_path, 'music.npy')
    micro_list_path = os.path.join(save_path, 'microphone_list.npy')
    covariance_matrix_path = os.path.join(save_path, 'covariance_matrix.npy')
    signals_path = os.path.join(save_path, 'signals.npy')

    # t = abs(signals).mean((0, 2), keepdims=True)
    maxFrame = abs(signals).mean((0, 2)).argmax()

    maxFrame_signal = signals[:, maxFrame, :].astype(np.float32)

    maxFrame_signal = maxFrame_signal / np.max(np.abs(maxFrame_signal))

    np.save(signals_path, maxFrame_signal)


def multi_processing_batch_processing_wave_from_txt_to_complex_spectrum(file_list: list, out_path,
                                                                        finish_num, lock):
    nfft = 512
    for file in file_list:
        file = file.replace('\\', '/')
        file_name = file.split('/')[-1].split('.')[0]

        for direction in directionEnum:
            direction_path = os.path.join(out_path, str(direction))

            lock.acquire()
            if os.path.exists(direction_path) is False:
                os.makedirs(direction_path)
            lock.release()

            save_path = os.path.join(direction_path, file_name)
            if not os.path.exists(save_path):
                # check = check_file_already_exist(file_name, direction_path)
                # if check:
                spectrum, music_numpy, microphone_list, covariance_matrix, signals = create_room_box(direction, file,
                                                                                                     nfft=nfft)
                os.makedirs(save_path)
                save_complex_spectrum_and_music(save_path=save_path, spectrum=spectrum, music=music_numpy,
                                                micro_list=microphone_list, covariance_matrix=covariance_matrix,
                                                signals=signals)
                # print('save +1')

            lock.acquire()
            finish_num.value += 1
            if finish_num.value % 100 == 0:
                print("finish" + str(finish_num.value))
            lock.release()

    print("finished")


def main(one_path1, one_path2):
    # one_path1 = 'G:\\soundID\\DOAdata\\test_wav_txt\\vad'
    # one_path2 = 'G:\\soundID\\DOAdata\\small_dataset\\s_m_numpy_5mm\\test'

    # 读取所有完成vad的从wav转化成txt的数据
    file_list = glob.glob(os.path.join(one_path1, '*'))

    # 取前2200条数据
    file_list = file_list[:2200]

    print(f"total files num:{len(file_list)}")

    threads = []
    thread_num = 20
    one_list_num = math.ceil(len(file_list) / thread_num)
    file_list_split = []
    for i in range(0, len(file_list), one_list_num):
        file_list_split.append(file_list[i:i + one_list_num])

    finish_num = Manager().Value('I', 0)
    lock = Lock()

    for i in range(thread_num):
        p = Process(target=multi_processing_batch_processing_wave_from_txt_to_complex_spectrum,
                    args=(file_list_split[i], one_path2, finish_num, lock))
        threads.append(p)
        print("thread: " + str(i) + " ready")
        # print("direction range:", end=' ')
        # print(",".join(str(x) for x in file_list_split[i]))
        print(' ')

    for p in threads:
        p.daemon = True
        p.start()

    for p in threads:
        p.join()
        print("finish two step")


if __name__ == '__main__':
    # all_time = time.time()
    # main(one_path1='F:\\soundID\\DOAdata\\train_wav_txt\\vad',
    #      one_path2='F:\\soundID\\DOAdata\\small_dataset\\s_m_numpy_5mm\\s_m_numpy_5mm_train1')
    # print(f'use time:{time.time() - all_time}')

    # 生成训练集
    all_time = time.time()
    in_path = '/home/work3/DOA_data/train_wav_txt/vad'
    out_path = '/home/ssd02/all_direction_oneFileAllDirection_onlySingleFrame_2200_0db_normalize_lowfilter_mic12x12/train'
    print('#####################################################')
    print(in_path)
    print('------------------------->')
    print(out_path)
    print('#####################################################')
    main(one_path1=in_path,
         one_path2=out_path)
    print(f'use time:{time.time() - all_time}')

    # 生成训练集
    # all_time = time.time()
    # in_path = '/home/work/TJ/data/train_wav_txt/vad'
    # out_path = '/home/ssd2/doa_data/all_direction_oneFileAllDirection_onlySingleFrame_2200_0db_no_preprocess_lowfilter_mic10x10/train'
    # print('#####################################################')
    # print(in_path)
    # print('------------------------->')
    # print(out_path)
    # print('#####################################################')
    # main(one_path1=in_path,
    #      one_path2=out_path)
    # print(f'use time:{time.time() - all_time}')

    # 生成测试集
    # all_time = time.time()
    # in_path = '/home/work/TJ/data/test_wav_txt/vad'
    # out_path = '/home/work/TJ/data/all_direction_oneFileAllDirection_onlySingleFrame_2200_0db_preprocess_low_filter/test'
    # print('#####################################################')
    # print(in_path)
    # print('------------------------->')
    # print(out_path)
    # print('#####################################################')
    # main(one_path1=in_path,
    #      one_path2=out_path)
    # print(f'use time:{time.time() - all_time}')

    # os.system('python produce_music_dataset.py')
