#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

"""
License: Apache 2.0 from Megvii
Modified by: Dat Tran
"""

import torch
import torch.nn as nn
import numpy as np


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2

        # find correct group value
        if groups is None:
            groups = 1

        if groups > 1:
            grp_value = None
            for grp in range(groups, 0, -1):
                if in_channels % grp == 0 and out_channels % grp == 0:
                    grp_value = grp
                    break
            if grp_value is None:
                grp_value = 1
        else:
            grp_value = 1

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=grp_value,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)
        if grp_value > 1:
            self.perm_indices = np.arange(out_channels)
            self.perm_indices = np.transpose(np.reshape(self.perm_indices, (grp_value, -1))).flatten().tolist()
        else:
            self.perm_indices = None

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        if self.perm_indices is not None:
            x = x[:, self.perm_indices, :, :]
        return x

    def fuseforward(self, x):
        x = self.act(self.conv(x))
        if self.perm_indices is not None:
            x = x[:, self.perm_indices, :, :]
        return x



class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu", bias=False):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
            bias=False,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act, bias=bias
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
        groups=1,
        bias=False,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        if groups is not None:
            self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act, groups=groups, bias=bias)
            self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act, groups=groups, bias=bias)
        else:
            self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act, bias=bias)
            self.conv2 = DWConv(hidden_channels, out_channels, 3, stride=1, act=act, bias=bias)

        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu", groups=1, bias=False,
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        if groups is not None:
            self.conv1 = BaseConv(
                in_channels, hidden_channels, 1, stride=1, act=activation, groups=groups, bias=bias
            )
        else:
            self.conv1 = DWConv(
                in_channels, hidden_channels, 1, stride=1, act=activation, bias=bias
            )

        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)

        self.conv2 = BaseConv(
            conv2_channels, out_channels, 1, stride=1, act=activation, groups=groups, bias=bias
        )

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
        groups=1,
        bias=False,
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act, groups=groups, bias=bias)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act, groups=groups, bias=bias)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act, groups=groups, bias=bias)

        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act, groups=groups, bias=bias
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu", groups=1, bias=False):
        super().__init__()
        if groups is not None:
            self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act, groups=groups, bias=bias)
        elif groups is None:
            self.conv = DWConv(in_channels * 4, out_channels, ksize, stride, act=act, bias=bias)
        else:
            raise RuntimeError("groups must in [1, 4, None] for the focus module")

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)
