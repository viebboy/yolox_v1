#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            #assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs


class YOLOXDeploy(nn.Module):
    """
    This model class is used to create a deployable model
    """

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))

    def load_weights_from(self, model):
        new_state_dict = model.state_dict()
        cur_state_dict = self.state_dict()
        count = 0
        total = 0
        for name in cur_state_dict.keys():
            total += 1
            if name in new_state_dict.keys():
                cur_state_dict[name] = new_state_dict[name].detach().clone()
                count += 1
            else:
                print(f'layer: {name} doesnt not appear in source model')
        self.load_state_dict(cur_state_dict)

        print(f'loading weights for {count} layers, there are {total} layers in total')
