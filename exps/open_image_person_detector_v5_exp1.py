#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # backbone params
        self.nb_init_filters = 32
        self.backbone_groups = [4, 4, 8, 8]
        self.fpn_groups = 4
        self.head_groups = 4
        self.stages = [3, 3, 3, 3]
        self.grow_rates = [1.5, 2.0, 2.5, 3.0]
        self.depth_in_fpn = 2
        # head params
        self.head_hidden_dim = 32

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
            from yolox.models.yoloxv5 import YOLOX, YOLOPAFPN, YOLOXHead
            backbone = YOLOPAFPN(
                nb_init_filters=self.nb_init_filters,
                backbone_groups=self.backbone_groups,
                fpn_groups=self.fpn_groups,
                stages=self.stages,
                grow_rates=self.grow_rates,
                depth_in_fpn=self.depth_in_fpn,
            )
            head = YOLOXHead(
                in_channels=backbone.in_channels,
                hidden_dim=self.head_hidden_dim,
                num_classes=self.num_classes,
                groups=self.head_groups,
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

        from yolox.models.yoloxv5 import YOLOXDeploy, YOLOPAFPN, YOLOXHeadDeploy
        backbone = YOLOPAFPN(
            nb_init_filters=self.nb_init_filters,
            backbone_groups=self.backbone_groups,
            fpn_groups=self.fpn_groups,
            stages=self.stages,
            grow_rates=self.grow_rates,
            depth_in_fpn=self.depth_in_fpn,
        )
        head = YOLOXHeadDeploy(
            input_height=self.test_size[0],
            input_width=self.test_size[1],
            in_channels=backbone.in_channels,
            hidden_dim=self.head_hidden_dim,
            num_classes=self.num_classes,
            groups=self.head_groups,
        )
        model = YOLOXDeploy(backbone, head)

        model.apply(init_yolo)
        model.head.initialize_biases(1e-2)
        return model

    def get_class_names(self):
        return ['Person',]
