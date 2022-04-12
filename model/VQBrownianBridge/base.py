import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def nonlinear(x):
    """非线性激活函数"""
    return x * torch.sigmoid(x)


def normalize(in_channels):
    """Group Normalization"""
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    """
        ResnetBlock: 通用的Resnet模块
    """

    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0):
        """
        :param in_channels: 输入数据的维度
        :param out_channels: 输出数据的维度
        :param conv_shortcut: 是否用卷积实现shortcut
        :param dropout: dropout
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # 当输入维度和输出维度不相同时，shortcut调整维度的方式，卷积或者mlp，1x1卷积的作用与线性层相同
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.lin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinear(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinear(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.lin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    """
        Attention block
    """

    def __init__(self, in_channels):
        """
        :param in_channels: 输入特征的维度
        """
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalize(in_channels)
        # 用1x1的卷积层代替线性层
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1).contiguous()
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1).contiguous()
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        return x + h_


class Downsample(nn.Module):
    """
        降采样层
    """

    def __init__(self, in_channels, with_conv=True):
        """
        :param in_channels: 输入特征的维度
        :param with_conv: 是否用卷积操作实现
        """
        super(Downsample, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            # 如果使用卷积实现，则需要进行填充
            pad = (0, 1, 0, 1)
            x = nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            # 如果不使用卷积实现，则利用平均池化层实现
            x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Upsample(nn.Module):
    """
        上采样层
    """

    def __init__(self, in_channels, with_conv):
        super(Upsample, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x