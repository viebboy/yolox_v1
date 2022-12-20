#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        nb_init_filters,
        groups,
        stages,
        grow_rates,
        depth_in_fpn,
        in_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(
            nb_init_filters=nb_init_filters,
            groups=groups,
            stages=stages,
            grow_rates=grow_rates,
            act=act,
        )
        self.in_features = in_features
        self.compute_in_channels()

        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            self.in_channels[2],
            self.in_channels[1],
            1,
            1,
            act=act,
            groups=groups,
        )

        self.C3_p4 = CSPLayer(
            self.in_channels[1] * 2,
            self.in_channels[1],
            depth_in_fpn,
            False,
            act=act,
            groups=groups,
        )  # cat

        self.reduce_conv1 = BaseConv(
            self.in_channels[1], self.in_channels[0], 1, 1, act=act, groups=groups,
        )
        self.C3_p3 = CSPLayer(
            self.in_channels[0]*2,
            self.in_channels[0],
            depth_in_fpn,
            False,
            act=act,
            groups=groups,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            self.in_channels[0], self.in_channels[0], 3, 2, act=act, groups=groups,
        )
        self.C3_n3 = CSPLayer(
            self.in_channels[0]*2,
            self.in_channels[1],
            depth_in_fpn,
            False,
            act=act,
            groups=groups,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            self.in_channels[1], self.in_channels[1], 3, 2, act=act, groups=groups,
        )
        self.C3_n4 = CSPLayer(
            self.in_channels[1]*2,
            self.in_channels[2],
            depth_in_fpn,
            False,
            act=act,
            groups=groups,
        )

    @torch.no_grad()
    def compute_in_channels(self):
        x = torch.randn(1, 3, 384, 384)
        out = self.backbone(x)
        self.in_channels = [out[x].size(1) for x in self.in_features]

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
