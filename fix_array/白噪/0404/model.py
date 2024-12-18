import torch
from torch import nn
import gc

# 结果交织5次,train和validation都用

# ------------------------- Epoch 24 / 2000 -------------------------
# train_loss:230.6496,train_acc:0.2152,train_bias_acc:0.8030,train_error:1.8019
# validation_loss:390.6510,validation_acc:0.1558,validation_bias_acc:0.6493,validation_error:3.0546


# 先过三个差分卷积，再做resnet
class DifferentialNet1(nn.Module):
    def __init__(self):
        super(DifferentialNet1, self).__init__()

        self.dropout = nn.Dropout3d(0.2)

        self.subDifferentialNet1 = SubDifferentialNet3(in_channel=514, out_channel=512)
        self.subDifferentialNet2 = SubDifferentialNet3(in_channel=512, out_channel=512)
        # self.subDifferentialNet3 = SubDifferentialNet3()

        # self.catConv = nn.Sequential(nn.Conv2d(in_channels=2 * 127, out_channels=2, kernel_size=1), nn.ReLU())

        self.afterSubSize = 6
        self.linear = nn.Sequential(
            nn.ReLU(),  # ==============
            nn.Dropout(),
            nn.Linear(256 * 2 * self.afterSubSize * self.afterSubSize,
                      256 * 2 * self.afterSubSize),
            nn.GroupNorm(32 * self.afterSubSize,
                         256 * 2 * self.afterSubSize),  # ============
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(256 * 2 * self.afterSubSize,
                      256 * 2),
            nn.GroupNorm(32,
                         256 * 2),  # ============
            nn.ReLU(inplace=True),
            nn.Linear(256 * 2, 360))

        self.sigmoid = nn.Sigmoid()

    # size = (b, 128, 2, m, n)
    def forward(self, real_imag):
        result = torch.zeros((real_imag.shape[0], 360), device=real_imag.device)
        occur_num = 5
        for _ in range(occur_num):
            x = self.dropout(real_imag)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.transpose(1, 2)
            x = x.reshape(x.shape[0], x.shape[1], 10, 10)
            x = self.subDifferentialNet2(self.subDifferentialNet1(x))
            x = torch.flatten(x, start_dim=1)
            x = self.linear(x)
            result += x
        result /= occur_num
        return self.sigmoid(result)


class SubDifferentialNet3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SubDifferentialNet3, self).__init__()
        self.in_channel_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3)))
        # self.concat_conv_real = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)

        self.filter = ResNet(num_res=1)

    # real and imag size = (b, c, m, n)
    def forward(self, real_imag):
        real_imag = self.in_channel_conv1(real_imag)

        real_imag = self.filter(real_imag)
        return real_imag


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


# 改过 和原始不同
class BasicBlock(nn.Module):
    def __init__(self, in_channels, inner_channel, out_channels=64, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, in_channels, stride)
        self.gn1 = nn.GroupNorm(in_channels // 8, in_channels)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(in_channels, inner_channel, stride)
        self.gn2 = nn.GroupNorm(in_channels // 8, inner_channel)
        # self.bn2 = nn.BatchNorm2d(out_channels)

        self.inner_channel = inner_channel
        self.out_channel = out_channels
        self.conv3 = nn.Conv2d(inner_channel, out_channels, kernel_size=1, stride=1)

        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        if self.inner_channel != self.out_channel:
            out = self.conv3(out)
        out += residual
        # out = self.gn2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_res=4):
        super(ResNet, self).__init__()
        self.res_sequential = nn.Sequential()
        for i in range(num_res):
            if i == num_res - 1:
                self.res_sequential.add_module(f'res_conv{(i + 1)}',
                                               BasicBlock(in_channels=512, inner_channel=256, out_channels=512))
            else:
                self.res_sequential.add_module(f'res_conv{(i + 1)}',
                                               BasicBlock(in_channels=512, inner_channel=256, out_channels=512))
                self.res_sequential.add_module(f'res_relu{(i + 1)}', nn.ReLU(inplace=True))  # =========

    def forward(self, src):
        src = self.res_sequential(src)
        return src
