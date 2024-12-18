import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
from torchsummary import summary
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from differentialNet import DifferentialNet1
from dataset import WavDataset
import numpy as np
import os
import time
import gc
import sys
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class contrastive_loss(nn.Module):
    def __init__(self):
        super(contrastive_loss, self).__init__()

    def forward(self, input, target, margin=1.25):
        input_x1 = torch.unsqueeze(input[0], 0)
        input_x2 = torch.unsqueeze(input[1], 0)
        target_x1 = target[0]
        target_x2 = target[1]
        if torch.equal(target_x1, target_x2):
            # if target_x1 == target_x2:
            y = 1
        else:
            y = 0
        distance = torch.pairwise_distance(input_x1, input_x2)
        return (1 - y) * 0.5 * distance + y * 0.5 * max(0, margin - distance)


# 单帧训练
def differential_train(net, opt, loss_fn, data_loader, device):
    net.train()
    total_loss = 0
    total_acc = 0
    total_error = 0
    total_bias_sum = 0

    total_signal_max = 0
    total_mix_max = 0
    total_signal_min = 0
    total_mix_min = 0
    total_max = - sys.maxsize - 1
    total_min = sys.maxsize
    total_max_2 = - sys.maxsize - 1
    total_min_2 = sys.maxsize

    for batch_ind, data in enumerate(data_loader):

        printProgressBar(batch_ind, data_loader.__len__())

        for i in range(0, len(data)):
            data[i] = data[i].to(device)
        # b, 128, 2, 100, 127
        all_frame_real_imag_data = data[0]
        # all_frame_real_imag_data = normalization(all_frame_real_imag_data)
        label = data[1]
        item_label = data[2]



        # total_signal_max += torch.mean(data[-4])
        # total_mix_max += torch.mean(data[-3])
        # total_signal_min += torch.mean(data[-2])
        # total_mix_min += torch.mean(data[-1])
        # print(batch_ind, total_signal_max / (batch_ind + 1), total_mix_max / (batch_ind + 1),
        #       total_signal_min / (batch_ind + 1), total_mix_min / (batch_ind + 1))
        # total_max = max(torch.max(data[-4]).item(), total_max)
        # total_max_2 = max(torch.max(data[-3]).item(), total_max_2)
        # total_min = min(torch.min(data[-2]).item(), total_min)
        # total_min_2 = min(torch.min(data[-1]).item(), total_min_2)

        # print('max', total_max)
        # print('min', total_min)
        # print('max_2', total_max_2)
        # print('min_2', total_min_2)

        iteration_loss = 0
        iteration_sum = 0
        iteration_error = 0
        iteration_bias_sum = 0

        # data_shape = all_frame_real_imag_data.shape
        # all_frame_real_imag_data = all_frame_real_imag_data.contiguous().view(data_shape[0], data_shape[1],
        #                                                                       data_shape[2], 10,
        #                                                                       10)

        opt.zero_grad()  # =======================

        # iteration_label = label[[i, i + 1]]
        # iteration_item_label = item_label[[i, i + 1]]
        # train_data = all_frame_real_imag_data[[i, i + 1]]
        all_frame_real_imag_data = Variable(all_frame_real_imag_data, requires_grad=True)
        out = net(all_frame_real_imag_data)

        x_grad_out = Variable(torch.ones_like(out), requires_grad=False)
        x_grad = autograd.grad(out, all_frame_real_imag_data, x_grad_out, create_graph=True, retain_graph=True,
                               only_inputs=True)[
            0]
        x_grad = x_grad.view(x_grad.shape[0], -1).pow(2).sum(1) ** 2
        x_div_gp = torch.mean(x_grad) * 10 / 2

        out_argmax = torch.max(out, dim=1)[1]
        one_acc = torch.eq(out_argmax, item_label)
        iteration_sum += (torch.sum(one_acc).detach().item()) / one_acc.shape[0]
        sub_error = torch.abs(out_argmax - item_label)
        temp_error = torch.min(sub_error, 360 - sub_error)

        loss = loss_fn(out, label) + x_div_gp + torch.sum(temp_error)

        loss.backward()
        opt.step()

        iteration_loss += loss.detach().item()

        iteration_error += torch.sum(temp_error).detach().item() / temp_error.shape[0]

        check_a = torch.ones(size=temp_error.shape, device=device)
        check_b = torch.zeros(size=temp_error.shape, device=device)
        bias_acc = torch.where(temp_error < 3, check_a, check_b)
        iteration_bias_sum += torch.sum(bias_acc).detach().item() / bias_acc.shape[0]

        total_loss += iteration_loss
        total_acc += iteration_sum
        total_error += iteration_error
        total_bias_sum += iteration_bias_sum

        gc.collect()
        del data, all_frame_real_imag_data, label
        gc.collect()

    # print('max', total_max)
    # print('min', total_min)
    # print('max_2', total_max_2)
    # print('min_2', total_min_2)
    # exit()
    return total_loss / len(data_loader), total_acc / len(data_loader), total_error / len(
        data_loader), total_bias_sum / len(data_loader)


def differential_validation(net, loss_fn, data_loader, device):
    net.eval()
    total_loss = 0
    total_acc = 0
    total_error = 0
    total_bias_sum = 0

    for batch_ind, data in enumerate(data_loader):
        for i in range(0, len(data)):
            data[i] = data[i].to(device)
        # b, 128, 2, 100, 127
        all_frame_real_imag_data = data[0]
        # all_frame_real_imag_data = normalization(all_frame_real_imag_data)
        label = data[1]
        item_label = data[2]

        iteration_loss = 0
        iteration_sum = 0
        iteration_error = 0
        iteration_bias_sum = 0

        all_frame_real_imag_data = Variable(all_frame_real_imag_data, requires_grad=True)
        out = net(all_frame_real_imag_data)

        x_grad_out = Variable(torch.ones_like(out), requires_grad=False)
        x_grad = autograd.grad(out, all_frame_real_imag_data, x_grad_out, create_graph=True, retain_graph=True,
                               only_inputs=True)[
            0]
        x_grad = x_grad.view(x_grad.shape[0], -1).pow(2).sum(1) ** 2
        x_div_gp = torch.mean(x_grad) * 10 / 2

        out_argmax = torch.max(out, dim=1)[1]
        one_acc = torch.eq(out_argmax, item_label)
        iteration_sum += (torch.sum(one_acc).detach().item()) / one_acc.shape[0]
        sub_error = torch.abs(out_argmax - item_label)
        temp_error = torch.min(sub_error, 360 - sub_error)

        loss = loss_fn(out, label) + x_div_gp + torch.sum(temp_error)

        iteration_loss += loss.detach().item()

        iteration_error += torch.sum(temp_error).detach().item() / temp_error.shape[0]

        check_a = torch.ones(size=temp_error.shape, device=device)
        check_b = torch.zeros(size=temp_error.shape, device=device)
        bias_acc = torch.where(temp_error < 3, check_a, check_b)
        iteration_bias_sum += torch.sum(bias_acc).detach().item() / bias_acc.shape[0]

        total_loss += iteration_loss
        total_acc += iteration_sum
        total_error += iteration_error
        total_bias_sum += iteration_bias_sum

        gc.collect()
        del data, all_frame_real_imag_data, label
        gc.collect()

    return total_loss / len(data_loader), total_acc / len(data_loader), total_error / len(
        data_loader), total_bias_sum / len(data_loader)


def train_loop(net, opt, loss_fn):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('gpu use:' + str(torch.cuda.is_available()))
    if device == 'cuda':
        print('device:' + str(torch.cuda.current_device()))
    else:
        print('device:cpu')
    batch_size = 1024
    num_worker = 32

    net.to(device)

    # 多gpu
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    # net = net.to(device)
    # net = nn.parallel.DistributedDataParallel(net, find_unused_parameters=True)

    # train_data = WavDataset(file_path='G:\\soundID\\DOAdata\\small_dataset\\s_m_numpy_5mm\\s_m_numpy_5mm_train1')
    # train_data = WavDataset(file_path='F:\\soundID\\DOAdata\\all_direction_oneFileAllDirection_onlySingleFrame_2200_0db_no_preprocess_lowfilter_mic10x10_brownnoise')
    train_data = WavDataset(file_path='/home/ssd2/doa_data/all_direction_oneFileAllDirection_onlySingleFrame_2200_0db_normalize_lowfilter_mic10x10_brownnoise/train')
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=num_worker, shuffle=True,
                              pin_memory=True)
    # num_work=16
    # 多gpu
    # train_data_sampler = DistributedSampler(train_data)
    # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=2,
    #                           sampler=train_data_sampler)

    # validation_data = WavDataset(file_path='G:\\soundID\\DOAdata\\small_dataset\\s_m_numpy_5mm\\s_m_numpy_5mm_train1')
    # validation_data = WavDataset(file_path='F:\\soundID\\DOAdata\\all_direction_oneFileAllDirection_onlySingleFrame_2200_0db_no_preprocess_lowfilter_mic10x10_brownnoise')
    validation_data = WavDataset(file_path='/home/ssd2/doa_data/all_direction_oneFileAllDirection_onlySingleFrame_2200_0db_normalize_lowfilter_mic10x10_brownnoise/validation')
    validation_loader = DataLoader(dataset=validation_data, batch_size=batch_size, num_workers=num_worker, shuffle=True,
                                   pin_memory=True)
    # 多gpu
    # validation_data_sampler = DistributedSampler(validation_data)
    # validation_loader = DataLoader(dataset=validation_data, batch_size=batch_size, num_workers=2,
    #                                sampler=validation_data_sampler)

    print("train")
    epochs = 2000
    path = '/home/work/TJ/DOA/save/pre/0227'
    for epoch in range(0, epochs):
        time1 = time.time()
        # epoch_loss, epoch_acc = 0, 0
        epoch_loss, epoch_acc, epoch_error, epoch_bias_acc = differential_train(net, opt, loss_fn, train_loader, device)
        if (epoch + 1) % 1 == 0:
            validation_loss, validation_acc, validation_error, validation_bias_acc = differential_validation(net,
                                                                                                             loss_fn,
                                                                                                             validation_loader,
                                                                                                             device)
            print("-" * 25, f"Epoch {epoch + 1} / {epochs}", "-" * 25)
            print(
                f'train_loss:{epoch_loss:.4f},train_acc:{epoch_acc:.4f},train_bias_acc:{epoch_bias_acc:.4f},train_error:{epoch_error:.4f}')
            print(
                f'validation_loss:{validation_loss:.4f},validation_acc:{validation_acc:.4f},validation_bias_acc:{validation_bias_acc:.4f},validation_error:{validation_error:.4f}')
            with open(os.path.join(path, 'train_loss.txt'), 'a') as f:
                f.write(str(epoch + 1) + ' ' + str(epoch_loss) + '\n')
            with open(os.path.join(path, 'train_acc.txt'), 'a') as f:
                f.write(str(epoch + 1) + ' ' + str(epoch_acc) + '\n')
            with open(os.path.join(path, 'train_bias_acc.txt'), 'a') as f:
                f.write(str(epoch + 1) + ' ' + str(epoch_bias_acc) + '\n')
            with open(os.path.join(path, 'train_error.txt'), 'a') as f:
                f.write(str(epoch + 1) + ' ' + str(epoch_error) + '\n')
            with open(os.path.join(path, 'validation_loss.txt'), 'a') as f:
                f.write(str(epoch + 1) + ' ' + str(validation_loss) + '\n')
            with open(os.path.join(path, 'validation_acc.txt'), 'a') as f:
                f.write(str(epoch + 1) + ' ' + str(validation_acc) + '\n')
            with open(os.path.join(path, 'validation_bias_acc.txt'), 'a') as f:
                f.write(str(epoch + 1) + ' ' + str(validation_bias_acc) + '\n')
            with open(os.path.join(path, 'validation_error.txt'), 'a') as f:
                f.write(str(epoch + 1) + ' ' + str(validation_error) + '\n')
        else:
            print("-" * 25, f"Epoch {epoch + 1} / {epochs}", "-" * 25)
            print(f'train_loss:{epoch_loss:.4f},train_acc:{epoch_acc:.4f},train_error:{epoch_error:.4f}')
            with open(os.path.join(path, 'train_loss.txt'), 'a') as f:
                f.write(str(epoch + 1) + ' ' + str(epoch_loss) + '\n')
            with open(os.path.join(path, 'train_acc.txt'), 'a') as f:
                f.write(str(epoch + 1) + ' ' + str(epoch_acc) + '\n')
            with open(os.path.join(path, 'train_bias_acc.txt'), 'a') as f:
                f.write(str(epoch + 1) + ' ' + str(epoch_bias_acc) + '\n')
            with open(os.path.join(path, 'train_error.txt'), 'a') as f:
                f.write(str(epoch + 1) + ' ' + str(epoch_error) + '\n')

        print(f'time:{time.time() - time1:.4f}')

        if (epoch + 1) % 5 == 0:
            torch.save(net.state_dict(), path + '/differential_net' + str(epoch + 1) + '.pth')


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    # print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\n')
    s = '{} |{}| {}% {}'.format(prefix, bar, percent, suffix)
    sys.stdout.write(' ' * (s.__len__() + 3) + '\r')
    sys.stdout.flush()
    sys.stdout.write(s + '\r')
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == '__main__':
    # transformer_model = FrameMicroTransFormerEncoder()  # 只使用编码器
    # # transformer_model.load_state_dict(torch.load('/home/work2/TJ/code/DOA/save/encode/12/space_net55.pth'))
    #
    # # print(transformer_model)
    #
    # opt = torch.optim.SGD(transformer_model.parameters(), lr=0.05, momentum=0.9)
    # # opt = torch.optim.Adam(transformer_model.parameters())
    #
    # # loss_fn = nn.CrossEntropyLoss()  # 单源
    # loss_fn = nn.BCELoss()  # 多源
    # train_loop(transformer_model, opt, loss_fn)

    # a = np.load('G:\\soundID\\DOAdata\\small_dataset\\s_m_numpy_5mm\\s_m_numpy_5mm_train\\0\\100-121669-0001\\complex_spectrum.npy')
    # real = np.real(a)
    # imag = np.imag(a)
    # test = np.concatenate((real, imag), axis=1)
    # print(real, imag)
    model = DifferentialNet1()
    # model.load_state_dict(torch.load('/home/work/TJ/DOA/save/music/differential/33kernel/real_imag_concat_singleFrame/2200_0307_addgrad_5db/differential_net430.pth'))
    print(model)
    # summary(model.cuda(), input_size=(514, 10, 10), batch_size=1024)
    opt = torch.optim.Adam(model.parameters(), lr=0.5 * 1e-4, betas=(0.5, 0.9))  # =========
    # opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.BCELoss()
    train_loop(model, opt, loss_fn)
