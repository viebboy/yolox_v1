#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch
import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.head_groups = 4
        self.head_hidden_dims = [48, 48, 48]
        self.use_bias = True

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


    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models.yoloxv9 import YOLOX, BackBone, YOLOXHead
            backbone = BackBone()
            with torch.no_grad():
                x = torch.rand(1, 3, self.test_size[0], self.test_size[1])
                fpn_outs = backbone(x)
                in_channels = [None,] + [item.shape[1] for item in fpn_outs] # None for 1st one because only 2 FPN

            head = YOLOXHead(
                in_channels=in_channels,
                hidden_dim=self.head_hidden_dims,
                num_classes=self.num_classes,
                groups=self.head_groups,
                nb_fpn=2, # pd backbone only has 2 fpn outputs
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

        from yolox.models.yoloxv9 import YOLOXDeploy, BackBone, YOLOXHeadDeploy
        backbone = BackBone()
        with torch.no_grad():
            x = torch.rand(1, 3, self.test_size[0], self.test_size[1])
            fpn_outs = backbone(x)
            in_channels = [None,] + [item.shape[1] for item in fpn_outs] # None for 1st one because only 2 FPN

        head = YOLOXHeadDeploy(
            input_height=self.test_size[0],
            input_width=self.test_size[1],
            in_channels=in_channels,
            hidden_dim=self.head_hidden_dims,
            num_classes=self.num_classes,
            groups=self.head_groups,
            nb_fpn=2,
            bias=self.use_bias,
        )
        model = YOLOXDeploy(backbone, head)

        model.apply(init_yolo)
        model.head.initialize_biases(1e-2)
        return model

    def get_class_names(self):
        return ['Person',]


exp = Exp()
model = exp.get_deploy_model()
x = torch.randn(1, 3, 384, 384)
y = model(x)
print(y.shape)
print(model)
