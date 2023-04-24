#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import torch.nn as nn
import torch
from yolox.exp import Exp as MyExp
import itertools
import hashlib
import json
import thop
from pprint import pprint


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        self.hyperparameter_names = [
            'nb_init_filters',
            'backbone_groups',
            'fpn_groups',
            'head_groups',
            'stages',
            'grow_rates',
            'depth_in_fpn',
            'head_hidden_dim',
        ]

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


    def set_hyperparameters(self, **kwargs):
        # set values
        for name, value in kwargs.items():
            if name in self.hyperparameter_names:
                if name == 'backbone_groups':
                    assert len(value) == 4
                elif name == 'stages':
                    assert len(value) == 4
                elif name == 'grow_rates':
                    assert len(value) == 4
                setattr(self, name, value)
            else:
                raise RuntimeError("Unknown hyperparameter name: {}".format(name))

    def get_config_name(self, config):
        return hashlib.sha256(dill.dumps(conf)).hexdigest()


    def create_factory(
        self,
        space_config,
        output_path,
        batch_size,
        opset_version,
        do_constant_folding,
    ):
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        for key in space_config:
            if key not in self.hyperparameter_names:
                raise RuntimeError(f'missing key: {key}')

        backbone_groups = list(itertools.product(*[space_config['backbone_groups']] * 4))
        stages = list(itertools.product(*[space_config['stages']] * 4))
        grow_rates = list(itertools.product(*[space_config['grow_rates']] * 4))
        all_ = list(
            itertools.product(
                space_config['nb_init_filters'],
                backbone_groups,
                space_config['fpn_groups'],
                space_config['head_groups'],
                stages,
                grow_rates,
                space_config['depth_in_fpn'],
                space_config['head_hidden_dim']
            )
        )
        configs = []
        for item in all_:
            configs.append(dict(zip(self.hyperparameter_names, item)))

        if batch_size is not None:
            dummy_input = torch.randn(batch_size, 3, self.test_size[0], self.test_size[1])
            dynamic_axes = None
        else:
            dummy_input = torch.randn(1, 3, self.test_size[0], self.test_size[1])
            dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

        input_names = ['input']
        output_names = ['output']

        for idx, conf in enumerate(configs):
            print(f'processing configuration index {idx} / {len(configs)}')
            name = self.get_config_name(conf)
            json_file = os.path.join(output_path, name + '.json')
            onnx_file = os.path.join(output_path, name + '.onnx')
            if os.path.exists(json_file) and os.path.exists(onnx_file):
                continue

            self.set_hyperparameters(**conf)

            model = self.get_deploy_model()
            model.eval()
            try:
                outputs = model(dummy_input)
                is_valid = True
            except Exception as e:
                print('invalid model config')
                pprint(conf, indent=2)
                is_valid = False

            if is_valid:
                try:
                    torch.onnx.export(
                        model,
                        dummy_input,
                        onnx_file,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                        verbose=False,
                        opset_version=opset_version,
                        do_constant_folding=do_constant_folding,
                    )
                    with open(json_file, 'w') as fid:
                        conf['version'] = 'v5'
                        macs, nb_params = thop.profile(model, inputs=(dummy_input, ))
                        conf['nb_macs'] = macs
                        conf['nb_parameters'] = nb_params
                        json.dump(conf, fid, indent=2)
                except Exception as e:
                    print('export onnx failed for the following configuration')
                    pprint(conf, indent=2)
                    print(e)

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
