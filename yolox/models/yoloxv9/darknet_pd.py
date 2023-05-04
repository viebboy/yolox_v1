# type: ignore
"""
From https://git.taservs.net/axon-research/Plate-Detection-Trainer/blob/d7e92e467b85d15dc57adb1b41bdcb081d81b5d0/models.py
"""
import logging
from typing import Dict
from typing import List
from typing import Tuple
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utils import helpers as h
from .utils.parse_config import parse_model_config
from .utils.utils_func import build_targets
from .utils.utils_func import build_targets_multilabel
from .utils.utils_func import to_cpu
# from plate_detection.task import plot_samples
logger = logging.getLogger()
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.5

default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'darknet_v1_anchor_free.cfg')

def _parse_yolo_module(module_def, hyper_params):
    print(f'-- module_def = {module_def}')
    print(f'-- hyper_params = {hyper_params}')
    anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
    # Extract anchors
    anchors = [int(x) for x in module_def['anchors'].split(',')]
    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
    anchors = [anchors[i] for i in anchor_idxs]
    num_classes = int(module_def['classes'])
    img_size = int(hyper_params['height'])
    if 'ignore_thres' in hyper_params:
        ignore_thres = float(hyper_params['ignore_thres'])
    else:
        print('-- this may be a backward compability issue, please check if you want to use ignore_thres=.7 for yolo model')
        ignore_thres = 0.7
    return anchors, num_classes, img_size, ignore_thres


def create_modules(module_defs, device) -> Tuple[Dict, nn.ModuleList]:
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                'conv_{}'.format(module_i),
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def['stride']),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(
                    'batch_norm_{}'.format(module_i), nn.BatchNorm2d(
                        filters, momentum=0.9, eps=1e-5,
                    ),
                )
            if module_def['activation'] == 'leaky':
                modules.add_module(
                    'leaky_{}'.format(module_i),
                    nn.LeakyReLU(0.1),
                )
            elif module_def['activation'] == 'relu':
                modules.add_module('relu_{}'.format(module_i), nn.ReLU6())
            elif module_def['activation'] == 'linear':
                True  # do nothing
            else:
                print('no definition for this layer')
                raise Exception

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module(
                    '_debug_padding_{}'.format(
                        module_i,
                    ), nn.ZeroPad2d((0, 1, 0, 1)),
                )
            maxpool = nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=int(
                    (kernel_size - 1) // 2,
                ),
            )
            modules.add_module('maxpool_{}'.format(module_i), maxpool)

        elif module_def['type'] == 'upsample':
            upsample = Upsample(
                scale_factor=int(
                    module_def['stride'],
                ),
                mode='nearest',
            )
            modules.add_module('upsample_{}'.format(module_i), upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module('route_{}'.format(module_i), EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[1:][int(module_def['from'])]
            modules.add_module('shortcut_{}'.format(module_i), EmptyLayer())

        elif module_def['type'] == 'yolo':
            anchors, num_classes, img_size, ignore_thres = _parse_yolo_module(
                module_def, hyperparams,
            )
            yolo_layer = YOLOLayer(
                anchors,
                num_classes,
                img_size,
                ignore_thres=ignore_thres,
            )
            modules.add_module('yolo_{}'.format(module_i), yolo_layer)
        elif module_def['type'] == 'yolox':
            # yolo_layer = h.YOLOXHeadNoStemNoClsConvNoRegLoss(
            modules.add_module('yolo_{}'.format(module_i), PlaceHolder())
        elif module_def['type'] == 'yolo_multilabel':
            anchors, num_classes, img_size, ignore_thres = _parse_yolo_module(
                module_def, hyperparams,
            )
            yolo_layer = YOLOLayer_multilabel(
                anchors, num_classes, img_size, ignore_thres=ignore_thres,
            )
            modules.add_module('yolo_{}'.format(module_i), yolo_layer)
        else:
            print('no definition for this layer')
            raise Exception
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    def __init__(self):
        super(EmptyLayer, self).__init__()

class PlaceHolder(nn.Module):
    def __init__(self):
        super(PlaceHolder, self).__init__()

    def forward(self, x):
        return x


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=320, ignore_thres=0.7):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = ignore_thres
        self.mse_loss = nn.MSELoss()
        # F.binary_cross_entropy enables adding weight for classes (can be used to emphasize readability)
        # nn.BCRLoss() enables adding weight for batch elements (can be used for hard-mining)
        # self.bce_loss = F.binary_cross_entropy
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 150
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size_x = 0  # grid size
        self.grid_size_y = 0  # grid size

    def compute_grid_offsets(self, grid_size_y, grid_size_x, cuda=True):
        self.grid_size_y, self.grid_size_x = grid_size_y, grid_size_x
        g_y, g_x = self.grid_size_y, self.grid_size_x
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size_x

        # Calculate offsets for each grid
        self.grid_x = torch.arange(g_x).repeat(
            g_y, 1,
        ).view([1, 1, g_y, g_x]).type(FloatTensor)

        self.grid_y = torch.arange(g_y).repeat(
            g_x, 1,
        ).t().view([1, 1, g_y, g_x]).type(FloatTensor)

        self.scaled_anchors = FloatTensor(
            [(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors],
        )
        self.anchor_w = self.scaled_anchors[:, 0:1].view(
            (1, self.num_anchors, 1, 1),
        )
        self.anchor_h = self.scaled_anchors[:, 1:2].view(
            (1, self.num_anchors, 1, 1),
        )

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        # LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        # ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size_y, grid_size_x = x.size(2), x.size(3)
        #        print(x.size())
        #        stop

        prediction = (
            x.view(
                num_samples,
                self.num_anchors,
                self.num_classes + 5,
                grid_size_y,
                grid_size_x,
            ) .permute(
                0,
                1,
                3,
                4,
                2,
            ) .contiguous()
        )
        # print(prediction.shape)
        # print(self.grid_size)

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size_x != self.grid_size_x:
            self.compute_grid_offsets(grid_size_y, grid_size_x, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with
            # conf. loss)
            loss_x = F.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = F.mse_loss(y[obj_mask], ty[obj_mask])
            # torch.log(torch.Tensor([((70 / 3840) * grid_size_x)]) / 0.625)
            loss_w = F.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = F.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = F.binary_cross_entropy(
                pred_conf[obj_mask], tconf[obj_mask],
            )
            loss_conf_noobj = F.binary_cross_entropy(
                pred_conf[noobj_mask], tconf[noobj_mask],
            )
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = F.binary_cross_entropy(
                pred_cls[obj_mask], tcls[obj_mask],
            )
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf = (pred_conf > CONF_THRESHOLD).float()
            #            conf50 = (pred_conf > 0.5).float()
            iou = (iou_scores > IOU_THRESHOLD).float()
            #            iou50 = (iou_scores > 0.5).float()
            #            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf * class_mask * tconf
            precision = torch.sum(iou * detected_mask) / (conf.sum() + 1e-16)
            #            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall = torch.sum(iou * detected_mask) / (obj_mask.sum() + 1e-16)
            #            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            #            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                'loss': to_cpu(total_loss).item(),
                'x': to_cpu(loss_x).item(),
                'y': to_cpu(loss_y).item(),
                'w': to_cpu(loss_w).item(),
                'h': to_cpu(loss_h).item(),
                'conf': to_cpu(loss_conf).item(),
                'cls': to_cpu(loss_cls).item(),
                'cls_acc': to_cpu(cls_acc).item(),
                'recall': to_cpu(recall).item(),
                #                "recall50": to_cpu(recall50).item(),
                #                "recall75": to_cpu(recall75).item(),
                'precision': to_cpu(precision).item(),
                'conf_obj': to_cpu(conf_obj).item(),
                'conf_noobj': to_cpu(conf_noobj).item(),
                'grid_size_x': grid_size_x,
                'grid_size_y': grid_size_y,
            }

            return output, total_loss


class YOLOLayer_multilabel(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=320, ignore_thres=0.7):
        super(YOLOLayer_multilabel, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = ignore_thres
        self.mse_loss = nn.MSELoss()
        # F.binary_cross_entropy enables adding weight for classes (can be used to emphasize readability)
        # nn.BCRLoss() enables adding weight for batch elements (can be used for hard-mining)
        # self.bce_loss = F.binary_cross_entropy
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 150
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(
            g, 1,
        ).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(
            g, 1,
        ).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor(
            [(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors],
        )
        self.anchor_w = self.scaled_anchors[:, 0:1].view(
            (1, self.num_anchors, 1, 1),
        )
        self.anchor_h = self.scaled_anchors[:, 1:2].view(
            (1, self.num_anchors, 1, 1),
        )

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        # ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)
        #        print(x.size())
        #        stop

        prediction = (
            x.view(
                num_samples,
                self.num_anchors,
                self.num_classes + 5,
                grid_size,
                grid_size,
            ) .permute(
                0,
                1,
                3,
                4,
                2,
            ) .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets_multilabel(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            # print('obj mask', obj_mask) # mask all anchor boxes
            # print('noobj mask', noobj_mask) # mask all anchor boxes
            # print('pred cls', pred_cls.shape) # prediction of all anchor
            # boxes
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(
                pred_conf[noobj_mask], tconf[noobj_mask],
            )
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            # weight = torch.ones_like(tcls[obj_mask])
            # read_weight = 1.7 * 2 / 2.7
            # non_read_weight = 2 / 2.7
            # weight[:, 0] = non_read_weight
            # weight[:, 1] = read_weight
            # loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask], weight)
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf = (pred_conf > CONF_THRESHOLD).float()
            #            conf50 = (pred_conf > 0.5).float()
            iou = (iou_scores > IOU_THRESHOLD).float()
            #            iou50 = (iou_scores > 0.5).float()
            #            iou75 = (iou_scores > 0.75).float()
            # For multilabel case, we do not need the condition of classifying correctly for a box to be considered
            # as a true positive. This is different from multiclass case, in which:
            # detected_mask = conf * class_mask * tconf
            detected_mask = conf * tconf
            precision = torch.sum(iou * detected_mask) / (conf.sum() + 1e-16)
            #            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall = torch.sum(iou * detected_mask) / (obj_mask.sum() + 1e-16)
            #            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            #            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                'loss': to_cpu(total_loss).item(),
                'x': to_cpu(loss_x).item(),
                'y': to_cpu(loss_y).item(),
                'w': to_cpu(loss_w).item(),
                'h': to_cpu(loss_h).item(),
                'conf': to_cpu(loss_conf).item(),
                'cls': to_cpu(loss_cls).item(),
                'cls_acc': to_cpu(cls_acc).item(),
                'recall': to_cpu(recall).item(),
                #                "recall50": to_cpu(recall50).item(),
                #                "recall75": to_cpu(recall75).item(),
                'precision': to_cpu(precision).item(),
                'conf_obj': to_cpu(conf_obj).item(),
                'conf_noobj': to_cpu(conf_noobj).item(),
                'grid_size': grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path=default_config, device='cuda', disabled_layers=[].copy()):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.width = int(self.module_defs[0]['width'])
        self.height = int(self.module_defs[0]['height'])
        self.disabled_layers = disabled_layers
        self.device = device
        _result = create_modules(self.module_defs, device)
        self.hyperparams: Dict = _result[0]
        self.module_list: nn.ModuleList = _result[1]

        self.yolo_layers = [
            layer[0] for layer in self.module_list if hasattr(
                layer[0], 'metrics',
            )
        ]

        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None, batch_id=0, verbose=False):
        input_batch = x
        img_dim = x.shape[-1]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(
                zip(self.module_defs, self.module_list),
        ):
            if module_def['type'] in self.disabled_layers:
                continue
            if module_def['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif module_def['type'] == 'route':
                # for layer_i, layer in enumerate(layer_outputs):
                #     print(layer_i, layer_outputs[int(layer_i)].shape)
                # print('-----')
                x = torch.cat(
                    [
                        layer_outputs[int(layer_i)]
                        for layer_i in module_def['layers'].split(',')
                    ], 1,
                )
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                yolo_layer: YOLOLayer = module[0]
                x, layer_loss = yolo_layer(x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            elif module_def['type'] == 'yolox':
                fpn_outs = [layer_outputs[-2], layer_outputs[-1]]

                # plot_samples(input_batch, targets)
                # targets[2:] = (x0, y0, w0, h0)
                """
                targets_ = targets[..., 1:].clone()
                h, w = input_batch.shape[2:]
                targets_[..., [1, 3]] *= w
                targets_[..., [2, 4]] *= h

                outputs = module[0](fpn_outs, targets_, input_batch)
                """
                yolo_outputs = fpn_outs
            elif module_def['type'] == 'yolo_multilabel':
                yolo_multilabel: YOLOLayer_multilabel = module[0]
                x, layer_loss = yolo_multilabel(x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        return yolo_outputs

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, 'rb') as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if 'darknet53.conv.74' in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(
                zip(self.module_defs, self.module_list),
        ):
            if i == cutoff:
                break
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b],
                    ).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b],
                    ).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b],
                    ).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b],
                    ).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b],
                    ).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w],
                ).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, 'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(
                zip(self.module_defs[:cutoff], self.module_list[:cutoff]),
        ):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
