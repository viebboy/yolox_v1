#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch
import torch.nn as nn
import copy

from yolox.exp import Exp as MyExp

DEFAULT_CONFIG = [
    {
        'type': 'conv-bn-act',
        'name': 'conv1',
        'input': 'input',
        'is_output': False,
        'in_channels': 3,
        'out_channels': 12,
        'kernel_size': 3,
        'stride': 2,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    # down sampling: 384 x 384 --> 192 x 192
    {
        'type': 'conv-bn-act',
        'name': 'conv2',
        'input': 'previous',
        'is_output': False,
        'in_channels': 12,
        'out_channels': 24,
        'kernel_size': 3,
        'stride': 2,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    # down sampling: 192 x 192 --> 96 x 96
    {
        'type': 'conv-bn-act',
        'name': 'conv3',
        'input': 'previous',
        'is_output': False,
        'in_channels': 24,
        'out_channels': 48,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv4',
        'input': 'previous',
        'is_output': False,
        'in_channels': 48,
        'out_channels': 24,
        'kernel_size': 1,
        'stride': 1,
        'padding': 0,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv5',
        'input': 'previous',
        'is_output': False,
        'in_channels': 24,
        'out_channels': 48,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'sum',
        'name': 'sum1',
        'input': ['conv3', 'conv5'],
        'is_output': False,
    },
    # down sampling: 96 x 96 --> 48 x 48
    {
        'type': 'pool2x2',
        'name': 'pool3',
        'input': 'previous',
        'is_output': False,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv6',
        'input': 'previous',
        'is_output': False,
        'in_channels': 48,
        'out_channels': 96,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv7',
        'input': 'previous',
        'is_output': False,
        'in_channels': 96,
        'out_channels': 48,
        'kernel_size': 1,
        'stride': 1,
        'padding': 0,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv8',
        'input': 'previous',
        'is_output': False,
        'in_channels': 48,
        'out_channels': 96,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'sum',
        'name': 'backbone_out_48x48',
        'input': ['conv6', 'conv8'],
        'is_output': True,
    },
    # down sampling: 48 x 48 --> 24 x 24
    {
        'type': 'pool2x2',
        'name': 'pool4',
        'input': 'previous',
        'is_output': False,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv9',
        'input': 'previous',
        'is_output': False,
        'in_channels': 96,
        'out_channels': 192,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv10',
        'input': 'previous',
        'is_output': False,
        'in_channels': 192,
        'out_channels': 96,
        'kernel_size': 1,
        'stride': 1,
        'padding': 0,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv11',
        'input': 'previous',
        'is_output': False,
        'in_channels': 96,
        'out_channels': 192,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'sum',
        'name': 'backbone_out_24x24',
        'input': ['conv9', 'conv11'],
        'is_output': True,
    },
    # down sampling: 24 x 24 --> 12 x 12
    {
        'type': 'pool2x2',
        'name': 'pool5',
        'input': 'previous',
        'is_output': False,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv12',
        'input': 'previous',
        'is_output': False,
        'in_channels': 192,
        'out_channels': 384,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv13',
        'input': 'previous',
        'is_output': False,
        'in_channels': 384,
        'out_channels': 192,
        'kernel_size': 1,
        'stride': 1,
        'padding': 0,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv14',
        'input': 'previous',
        'is_output': False,
        'in_channels': 192,
        'out_channels': 384,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'sum',
        'name': 'backbone_out_12x12',
        'input': ['conv12', 'conv14'],
        'is_output': True,
    }
]



class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.groups = 1
        self.use_bias = True
        self.backbone_dims = [12, 24, 48, 24, 48, 96, 48, 96, 192, 96, 192, 384, 192, 384]
        self.head_hidden_dims = [(48, 48, 48), (48, 48, 48), (48, 48, 48)]
        self.nb_fpn = 3

        #
        self.num_classes = 1
        self.input_size = (384, 384)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (384, 384)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False

        # Define yourself dataset path
        self.data_dir = None
        self.train_ann = None
        self.val_ann = None
        self.use_cache = False
        self.nb_shard = 32

        self.backbone_config = copy.deepcopy(DEFAULT_CONFIG)
        count = 0
        in_channels = 3
        for layer_idx, layer in enumerate(self.backbone_config):
            if layer['type'] == 'conv-bn-act':
                self.backbone_config[layer_idx]['groups'] = self.groups
                self.backbone_config[layer_idx]['bias'] = self.use_bias
                self.backbone_config[layer_idx]['in_channels'] = in_channels
                self.backbone_config[layer_idx]['out_channels'] = self.backbone_dims[count]
                in_channels = layer['out_channels']
                count += 1


    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models.yoloxv11 import YOLOX, VanillaCNN, YOLOXHead
            backbone = VanillaCNN(self.backbone_config)
            with torch.no_grad():
                x = torch.rand(1, 3, self.test_size[0], self.test_size[1])
                fpn_outs = backbone(x)
                in_channels = [item.shape[1] for item in fpn_outs]

            head = YOLOXHead(
                in_channels=in_channels,
                hidden_dims=self.head_hidden_dims,
                num_classes=self.num_classes,
                groups=self.groups,
                nb_fpn=self.nb_fpn,
                bias=self.use_bias,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_deploy_model(self):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        from yolox.models.yoloxv11 import YOLOXDeploy, VanillaCNN, YOLOXHeadDeploy
        backbone = VanillaCNN(self.backbone_config)
        with torch.no_grad():
            x = torch.rand(1, 3, self.test_size[0], self.test_size[1])
            fpn_outs = backbone(x)
            in_channels = [item.shape[1] for item in fpn_outs]

        head = YOLOXHeadDeploy(
            input_height=self.test_size[0],
            input_width=self.test_size[1],
            in_channels=in_channels,
            hidden_dims=self.head_hidden_dims,
            num_classes=self.num_classes,
            groups=self.groups,
            nb_fpn=self.nb_fpn,
            bias=self.use_bias,
        )
        model = YOLOXDeploy(backbone, head)
        model.apply(init_yolo)
        model.head.initialize_biases(1e-2)
        return model

    def get_class_names(self):
        return ['Person',]


if __name__ == '__main__':
    exp = Exp()
    model = exp.get_deploy_model()
    x = torch.randn(1, 3, 384, 384)
    y = model(x)
    print(y.shape)
    print(model)
