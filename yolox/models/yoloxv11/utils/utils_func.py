# type: ignore
"""
From https://git.taservs.net/axon-research/Plate-Detection-Trainer/blob/d7e92e467b85d15dc57adb1b41bdcb081d81b5d0/utils/utils_func.py
"""
import json
import logging
import math
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from joblib import delayed
from joblib import Parallel
from torch.autograd import Variable

logger = logging.getLogger()


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, 'r')
    names = fp.read().split('\n')[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    #    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    #    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    pad_x = 0
    pad_y = 0

    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
        Reference: https://axon.quip.com/qb6QA2n1M1N7/ALPR-Plate-Detection-Development#PAZACA0K56I
        For multi-class, this returns 4x5 matrix where
            Rows: val_precision, val_recall, val_map, val_f1
            Cols:
                1st: non-readable plates
                2nd: readable plates
                3rd: all plates (don't care about label)
                4th:

    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    pr_curves = dict()

    for c in tqdm.tqdm(unique_classes, desc='Computing AP'):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            if c == 1:  # Milder penanty for readable FPs
                # A readable prediction box that overlaps with a non-readable ground-truth box is
                # treated as a true positive. Normally, this is counted as a
                # false positive
                n_gt_plus = (tp[i] == (c * 2 + 1)).sum()
                n_gt += n_gt_plus
                tp_tmp = np.where(tp[i] >= (c * 2 + 1), 1, 0)
            else:
                tp_tmp = np.where(tp[i] == (c * 2 + 2), 1, 0)
            fpc = (1 - tp_tmp).cumsum()
            tpc = (tp_tmp).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            if len(recall_curve) < 1:
                print(f'empty recall curve for class {c}')
                continue
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))
            pr_curves[c] = (recall_curve, precision_curve)

    # Compute agnostic precision, recall, ap, f1
    n_gt = len(target_cls)
    n_p = len(pred_cls)
    if n_p == 0 or n_gt == 0:
        ap.append(0)
        r.append(0)
        p.append(0)
        cls_agnostic_pr_curves = (None, None)
    else:
        tp_tmp = np.where(tp > 0, 1, 0)
        fpc = (1 - tp_tmp).cumsum()
        tpc = (tp_tmp).cumsum()
        recall_curve = tpc / (n_gt + 1e-16)
        r.append(recall_curve[-1])
        precision_curve = tpc / (tpc + fpc)
        p.append(precision_curve[-1])
        ap.append(compute_ap(recall_curve, precision_curve))

        # record class-agnostic precision and recall curves
        cls_agnostic_pr_curves = (recall_curve, precision_curve)

    # Compute precision, recall, f1 for pred_boxes that match gt_boxes
    # pred_boxes that match gt_boxes are assigned one of the values 1,2,3,4
    # in true_positives variable in function get_batch_statistics.
    # true_positives is named tp in this function.
    i = tp > 0  # Filter out false positive detection boxes
    tp_detected = tp[i]

    for c in unique_classes:
        # FP. For example, a should-be nonreadable pred_box is classified as a
        # readable pred_box
        i1 = tp_detected == c * 2 + 1
        # TP. For example, a should-be readable pred_box is classified as a
        # readable pred_box
        i2 = tp_detected == c * 2 + 2
        # TP + FP for class c (e.g. total pred_boxes that are classified as
        # nonreadable)
        i3 = i1 ^ i2
        i4 = ~i3  # preds of other class
        # FP of other class is FN of this class. If a pred_box is classified as
        # nonreadable while its gt_label is readable, there is a false positive for nonreadable class
        # and a false negative for readable class
        i5 = (tp_detected[i4] % 2 == 1)
        n_gt = i5.sum() + i2.sum()  # TP + FN
        # Re-assign label from 1-4 to either 1 or 0 based on classification
        tp_tmp = np.where(tp_detected[i3] % 2, 0, 1)
        fpc = (1 - tp_tmp).cumsum()
        tpc = (tp_tmp).cumsum()
        # Recall
        recall_curve = tpc / (n_gt + 1e-16)
        if len(recall_curve) < 1:
            print(f'-- empty recall curve for class {c}')
            continue
        r.append(recall_curve[-1])

        # Precision
        precision_curve = tpc / (tpc + fpc)
        p.append(precision_curve[-1])

        # AP from recall-precision curve
        ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32'), pr_curves


def ap_per_class_multilabel(tp, conf, pred_cls, target_cls, cls_threshold=0.5):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    # logging.debug('preds: {}'.format(len(pred_cls)))
    # logging.debug('tps: {}'.format(tp.sum()))
    # logging.debug('target_cls: {}'.format(len(target_cls), target_cls[:5]))
    target_cls = np.array(target_cls).astype(np.int)
    unique_classes = np.unique(target_cls)
    target_onehot = np.eye(len(unique_classes))[target_cls]
    target_onehot[:, 0] = 1
    # logging.debug('target_onehot[:5, :]: {}'.format(target_onehot[:5, :]))
    # Find unique classes

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    pr_curves = dict()
    for c_index, c in enumerate(unique_classes):
        i = pred_cls[:, c_index] > cls_threshold
        # Number of ground truth objects
        n_gt = (target_onehot[:, c_index] == 1).sum()
        n_p = i.sum()  # Number of predicted objects
        # logging.debug('n_p for class {}: {}'.format(c_index, n_p))
        # logging.debug('n_gt for class {}: {}'.format(c_index, n_gt))
        if n_p == 0 and n_gt == 0:
            # logging.debug('no gts and no preds')
            continue
        elif n_p == 0 or n_gt == 0:
            if n_p == 0:
                logging.debug('no preds for: {}'.format(c))
            if n_gt == 0:
                logging.debug('no gts')
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            if c == 1:  # Milder penalty for readable FPs
                # A readable prediction box that is overlapped with a non-readable ground-truth box is
                # treated as a true positive. Normally, this is counted as a
                # false positive
                n_gt_plus = (tp[i, c_index] == 1).sum()
                n_gt += n_gt_plus
                tp_tmp = np.where(tp[i, c_index] >= 1, 1, 0)
            else:
                tp_tmp = tp[i, c_index]
            fpc = (1 - tp_tmp).cumsum()
            tpc = (tp_tmp).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))
            pr_curves[c] = (recall_curve, precision_curve)

    # Agnostic precision, recall, ap are the same as nonreadable metrics in
    # multilabel.
    p.append(p[0])
    r.append(r[0])
    ap.append(ap[0])
    # Compute precision, recall, f1 for pred_boxes that match gt_boxes

    i = (tp[:, 1] != 0)
    tp_detected = tp[i, 1]
    # In multilabel, because all gt boxes are labeled nonreadable, and all pred boxes are classified as nonreadable,
    # when we only consider pred_boxes that already match gt_boxes, all of these pred_boxes are correctly classified
    # Therefore, precision, recall, ap for nonreadable class is all 1.
    p.append(1.0)
    r.append(1.0)
    ap.append(1.0)
    # Calculating classification score for Readable class
    # FP. For example, a should-be nonreadable pred_box is classified as a
    # readable pred_box
    i1 = tp_detected == 1
    # TP. For example, a should-be readable pred_box is classified as a
    # readable pred_box
    i2 = tp_detected == 2
    i3 = i1 ^ i2  # TP + FP for class c
    # FN. For example, a should-be readable pred_box is classified as a
    # nonreadable pred_box
    i5 = tp_detected == -1
    n_gt = i2.sum() + i5.sum()  # TP + FN
    # Re-assign label from (-1,0,1,2) to either 1 or 0 based on classification
    tp_tmp = np.where(tp_detected[i3] % 2, 0, 1)
    fpc = (1 - tp_tmp).cumsum()
    tpc = (tp_tmp).cumsum()
    # Recall
    recall_curve = tpc / (n_gt + 1e-16)
    r.append(recall_curve[-1])

    # Precision
    precision_curve = tpc / (tpc + fpc)
    p.append(precision_curve[-1])

    # AP from recall-precision curve
    ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32'), pr_curves


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(
        outputs,
        targets,
        iou_threshold,
        small_boxes_filter=False,
):
    FRAME_HEIGHT_4K = 2160  # pixels
    MIN_PLATE_HEIGHT = 24  # pixels

    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]
        i = np.argsort(-pred_scores)
        pred_boxes, pred_scores, pred_labels = pred_boxes[i], pred_scores[i], pred_labels[i]

        # Init an array that determine a pred_box is a true positive (assigned
        # 1-4) or a false positive (assigned 0)
        true_positives = np.zeros(pred_boxes.shape[0])

        # annotations = targets[targets[:, 0] == sample_i][:, 1:]
        annotations = targets[sample_i][:, 1:]
        annotations = annotations[annotations[:, 2] > 1e-5]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            if small_boxes_filter:
                # areas = (target_boxes[:,2]-target_boxes[:,0])* (target_boxes[:,3]-target_boxes[:,1])/ (384 ** 2)
                # qualified = areas > (36 * 24) / (3840 * 2160)

                height = (target_boxes[:, 3] - target_boxes[:, 1]) / 384
                qualified = height > MIN_PLATE_HEIGHT / FRAME_HEIGHT_4K

                target_boxes = target_boxes[qualified]
                target_labels = target_labels[qualified]

                assert (target_boxes.shape[0] == len(target_labels))

            if len(target_boxes):
                for pred_i, (pred_box, pred_label) in enumerate(
                        zip(pred_boxes, pred_labels),
                ):

                    # If all target boxes are found, break
                    # remained pred_boxes are treated as false positives -
                    # assigned 0 in true_positives array
                    if len(detected_boxes) == len(annotations):
                        break

                    # Normally, if pred_label is not in target_labels, we treat this pred_box as a false positive
                    # However, we want to calculate agnostic precision, recall, ap, f1 and classification metrics in
                    # function ap_per_class. So even if pred_label is not in target_labels, we still calculate IoU and
                    # assign 1-4 (for later use in function get_batch_statistics). Do not uncomment the code below.
                    # Ignore if label is not one of the target labels
                    # if pred_label not in target_labels:
                    #     continue
                    ious = bbox_iou(pred_box.unsqueeze(0), target_boxes)
                    iou, box_index = ious.max(0)
                    if iou >= iou_threshold and box_index not in detected_boxes:
                        # arrange tp score depends on the type of matching & readability.
                        # this tp score will be used in ap_per_class
                        # tp = 0 => gt and pr has no iou match
                        # tp = 1 => gt readable, pr nonread
                        # tp = 2 => gt nonread, pr nonread
                        # tp = 3 => gt nonread, pr read
                        # tp = 4 => gt read, pr read
                        if pred_label == target_labels[box_index]:
                            true_positives[pred_i] = pred_label * 2 + 2
                        else:
                            true_positives[pred_i] = pred_label * 2 + 1
                        detected_boxes += [box_index]
        batch_metrics.append(
            ([true_positives, pred_scores, pred_labels], target_labels),
        )
    return batch_metrics


def get_batch_statistics_multilabel(
        outputs,
        targets,
        iou_threshold,
        cls_threshold=0.5,
):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    count_boxes = 0
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_objectness = output[:, 4]
        pred_class_scores = output[:, 5:-1]
        i = np.argsort(-pred_objectness)
        pred_boxes, pred_objectness, pred_class_scores = pred_boxes[
            i
        ], pred_objectness[i], pred_class_scores[i]
        n_labels = output.shape[1] - 5
        # Init an array with n_labels columns (e.g. 2 in this case) and len(pred_boxes) rows
        # Each column encodes for each label (nonreadable or readable)
        # Each row encodes for each pred_box
        # The values for this array are in (-1, 0, 1, 2) and will be explained
        # below
        true_positives = np.zeros((pred_boxes.shape[0], n_labels))

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        count_boxes += len(annotations)
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, pred_box in enumerate(pred_boxes):

                # If all target boxes are found, break
                # remained pred_boxes are treated as false positives - assigned
                # 0 in true_positives array
                if len(detected_boxes) == len(annotations):
                    break

                # Normally, if pred_label is not in target_labels, we treat this pred_box as a false positive
                # However, we want to calculate agnostic precision, recall, ap, f1 and classification metrics in
                # function ap_per_class. So even if pred_label is not in target_labels, we still calculate IoU and
                # assign 1-4 (for later use in function get_batch_statistics). Do not uncomment the code below.
                # Ignore if label is not one of the target labels
                # if pred_label not in target_labels:
                #     continue

                iou, box_index = bbox_iou(
                    pred_box.unsqueeze(0), target_boxes,
                ).max(0)
                if box_index in detected_boxes:
                    logging.debug(
                        'second box match one gt: {}'.format(
                            target_boxes[box_index],
                        ),
                    )
                if iou >= iou_threshold and box_index not in detected_boxes:
                    # arrange tp score depends on the type of matching & readability.
                    # this tp score will be used in ap_per_class_multilabel
                    # For Column 0-th:
                    # tp = 0 => gt and pr has no iou match
                    # tp = 1 => gt and pr has iou match
                    # For Column 1-st:
                    # tp = 0 => gt and pr has no iou match
                    # tp = 1 => gt nonread, pr read
                    # tp = 2 => gt read, pr read
                    # tp = -1 => gt read, pr nonread
                    true_positives[pred_i, 0] = 1
                    if target_labels[box_index] == 1 and pred_class_scores[
                        pred_i,
                        1,
                    ] > cls_threshold:
                        true_positives[pred_i, 1] = 2
                    elif pred_class_scores[pred_i, 1] > cls_threshold:
                        true_positives[pred_i, 1] = 1
                    elif target_labels[box_index] == 1:
                        true_positives[pred_i, 1] = -1
                    detected_boxes += [box_index]

        batch_metrics.append(
            [true_positives, pred_objectness, pred_class_scores],
        )
    return batch_metrics


def digitize(inp, _val: int):
    if isinstance(inp, pd.Series):
        return ((inp // _val * _val) + _val // 2).astype(int)
    return list((np.array(inp, int) // _val * _val) + _val // 2)


def get_digitize_vals(height: int, width: int) -> dict:
    return {
        'area': height * width * 4 or 800,
        'h': height // 2 or 12,
        'w': width // 2 or 8,
    }


def filter_small_or_large_plates(
        df: pd.DataFrame,
        height_keep: int,
        height_max: int,
        width_keep: int,
        width_max: int,
) -> pd.DataFrame:
    c0 = (height_keep <= df.gt_plate_height) & \
         (df.gt_plate_height <= height_max)
    c1 = (width_keep <= df.gt_plate_width) & \
         (df.gt_plate_width <= width_max)

    # c2 = (height_keep <= det_ann.pr_plate_height) & \
    #      (det_ann.pr_plate_height <= height_max)
    # c3 = (width_keep <= det_ann.pr_plate_width) & \
    #      (det_ann.pr_plate_width <= width_max)

    return df.loc[c0 & c1]


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[
            :,
            0
        ], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[
            :,
            0
        ], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.2):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat(
            (image_pred[:, :5], class_confs.float(), class_preds.float()), 1,
        )
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(
                detections[0, :4].unsqueeze(
                    0,
                ), detections[:, :4],
            ) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and
            # matching labels
            invalid = large_overlap  # & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (
                weights * detections[invalid, :4]
            ).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def non_max_suppression_multilabel(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        # logging.debug('image_pred[:, 4:]: {}'.format(image_pred[:, 4:]))
        # logging.debug('class_confs: {}'.format(class_confs))
        # logging.debug('class_preds: {}'.format(class_preds))
        detections = torch.cat((image_pred, class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(
                detections[0, :4].unsqueeze(
                    0,
                ), detections[:, :4],
            ) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and
            # matching labels
            invalid = large_overlap  # & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (
                weights * detections[invalid, :4]
            ).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def build_targets(
        pred_boxes,
        pred_cls,
        target,
        anchors,
        ignore_thres,
        use_iou_as_score=1,
):
    EPSILON = 1e-5

    # If train on multiple GPUs, batch indices of target will be mistaken.
    # For example, if we set batch_size = 64 and n_gpu = 2, values of the first column of target will be
    # from 0-31 but we want it to be from 0-63. So transferring it to cpu for calculation
    # and transfer it back to cuda at the end of this function.
    is_cuda = 1 if pred_boxes.is_cuda else 0

    new_targets = None
    for bi in range(target.shape[0]):
        if new_targets is None:
            new_targets = target[bi, target[bi, :, 3] > EPSILON]
            new_targets[:, 0] = bi
        else:
            tmp_target = target[bi, target[bi, :, 3] > EPSILON]
            tmp_target[:, 0] = bi
            new_targets = torch.cat([new_targets, tmp_target])
    # print(f'-- new target {new_targets.shape}', new_targets)
    target = new_targets
    # if is_cuda:
    #     pred_boxes, pred_cls, target, anchors = \
    #         pred_boxes.to(torch.device('cpu')), pred_cls.to(torch.device('cpu')), \
    #         target.to(torch.device('cpu')), anchors.to(torch.device('cpu'))

    BoolTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nGy = pred_boxes.size(2)
    nGx = pred_boxes.size(3)

    # Output tensors
    obj_mask = BoolTensor(nB, nA, nGy, nGx).fill_(0)
    noobj_mask = BoolTensor(nB, nA, nGy, nGx).fill_(1)
    class_mask = FloatTensor(nB, nA, nGy, nGx).fill_(0)
    iou_scores = FloatTensor(nB, nA, nGy, nGx).fill_(0)
    tx = FloatTensor(nB, nA, nGy, nGx).fill_(0)
    ty = FloatTensor(nB, nA, nGy, nGx).fill_(0)
    tw = FloatTensor(nB, nA, nGy, nGx).fill_(0)
    th = FloatTensor(nB, nA, nGy, nGx).fill_(0)
    tcls = FloatTensor(nB, nA, nGy, nGx, nC).fill_(0)

    # Convert to position relative to box
    target[:, [2, 4]] *= nGx
    target[:, [3, 5]] *= nGy
    target_boxes = target[:, 2:6]
    # logging.debug('target: {}'.format(target.shape, target))
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # logging.debug('ious {}'.format(ious.shape))
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()

    # Problem solved:
    # https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/157
    gj = torch.clamp(gj, 0, noobj_mask.size()[2] - 1)
    gi = torch.clamp(gi, 0, noobj_mask.size()[3] - 1)
    # Set masks
    # logging.debug('index: {}'.format([b, best_n, gj, gi]))
    # logging.debug(obj_mask.shape)
    obj_mask[b, best_n, gj, gi] = 1
    # logging.debug('obj_mask[b, best_n, gj, gi].shape {}'.format(obj_mask[b,
    # best_n, s gj, gi].shape))
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # logging.debug('tcls for multi-class\n: {}'.format(tcls[b, best_n, gj, gi, :5]))
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (
        pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels
    ).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(
        pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False,
    )

    if not use_iou_as_score:
        tconf = obj_mask.float()
    else:
        tconf = FloatTensor(obj_mask.shape).fill_(0)
        tconf[b, best_n, gj, gi] = best_ious

    # transfer back to gpu
    res = [
        iou_scores,
        class_mask,
        obj_mask,
        noobj_mask,
        tx,
        ty,
        tw,
        th,
        tcls,
        tconf,
    ]
    if is_cuda:
        res = [x.to(torch.device('cuda')) for x in res]

    return res


def build_targets_multilabel(
        pred_boxes,
        pred_cls,
        target,
        anchors,
        ignore_thres,
        use_iou_as_score=1,
):
    # If train on multiple GPUs, batch indices of target will be mistaken.
    # For example, if we set batch_size = 64 and n_gpu = 2, values of the first column of target will be
    # from 0-31 but we want it to be from 0-63. So transferring it to cpu for calculation
    # and transfer it back to cuda at the end of this function.
    is_cuda = 1 if pred_boxes.is_cuda else 0
    if is_cuda:
        pred_boxes, pred_cls, target, anchors = pred_boxes.to(
            torch.device('cpu'),
        ), pred_cls.to(
            torch.device('cpu'),
        ), target.to(
            torch.device('cpu'),
        ), anchors.to(
            torch.device('cpu'),
        )
    BoolTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if pred_boxes.is_cuda else torch.LongTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = BoolTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = BoolTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG, nC).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    # logging.debug('target: {}'.format(target.shape, target))
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    # logging.debug('index: {}'.format([b, best_n, gj, gi]))
    # logging.debug(obj_mask.shape)
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    # logging.debug('target_labels before\n: {}'.format(target_labels[:5])) # class labels for each sample
    # logging.debug(target_labels.shape)
    target_onehot = torch.FloatTensor(target_labels.shape[0], nC)
    target_onehot.zero_()
    target_onehot.scatter_(1, target_labels.unsqueeze(-1), 1)
    target_onehot[:, 0] = 1
    # logging.debug('target_onehot after\n: {}'.format(target_onehot[:5]))
    # logging.debug(target_onehot.shape)
    # logging.debug('tcls before\n: {}'.format(tcls[b, best_n, gj, gi, :5]))
    tcls[b, best_n, gj, gi, :] = target_onehot
    # logging.debug('tcls for multi-label\n: {}'.format(tcls[b, best_n, gj, gi, :5])) # assign class labels to ground truth tensor
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (
        pred_cls[b, best_n, gj, gi] == target_onehot
    ).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(
        pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False,
    )

    if not use_iou_as_score:
        tconf = obj_mask.float()
    else:
        tconf = FloatTensor(obj_mask.shape).fill_(0)
        tconf[b, best_n, gj, gi] = best_ious
    # transfer back to gpu
    res = []
    for _ in [
        iou_scores,
        class_mask,
        obj_mask,
        noobj_mask,
        tx,
        ty,
        tw,
        th,
        tcls,
        tconf,
    ]:
        res.append(_.to(torch.device('cuda')))
    return res
    # return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th,
    # tcls, tconf


def get_preproc_config_from_args(args):
    return {
        'rotate_max': args.rotate_max,
        'gamma_max': args.gamma_max,
        'gauss_min': args.gauss_min,
        'gauss_max': args.gauss_max,
    }


def bb_convert(x1, y1, x2, y2, x3, y3, x4, y4):
    left = min([x1, x2, x3, x4])
    right = max([x1, x2, x3, x4])
    top = min([y1, y2, y3, y4])
    bot = max([y1, y2, y3, y4])

    return left, right, top, bot


def correct(coord):
    x1, y1 = coord[0]['x'], coord[0]['y']
    x2, y2 = coord[1]['x'], coord[1]['y']
    x3, y3 = coord[2]['x'], coord[2]['y']
    x4, y4 = coord[3]['x'], coord[3]['y']
    left, right, top, bot = bb_convert(x1, y1, x2, y2, x3, y3, x4, y4)

    d1 = (x1 - left) ** 2 + (y1 - top) ** 2
    d2 = (x2 - left) ** 2 + (y2 - top) ** 2
    d3 = (x3 - left) ** 2 + (y3 - top) ** 2
    d4 = (x4 - left) ** 2 + (y4 - top) ** 2
    _1 = np.argmin([d1, d2, d3, d4])

    d1 = (x1 - right) ** 2 + (y1 - top) ** 2
    d2 = (x2 - right) ** 2 + (y2 - top) ** 2
    d3 = (x3 - right) ** 2 + (y3 - top) ** 2
    d4 = (x4 - right) ** 2 + (y4 - top) ** 2
    _2 = np.argmin([d1, d2, d3, d4])

    d1 = (x1 - right) ** 2 + (y1 - bot) ** 2
    d2 = (x2 - right) ** 2 + (y2 - bot) ** 2
    d3 = (x3 - right) ** 2 + (y3 - bot) ** 2
    d4 = (x4 - right) ** 2 + (y4 - bot) ** 2
    _3 = np.argmin([d1, d2, d3, d4])

    d1 = (x1 - left) ** 2 + (y1 - bot) ** 2
    d2 = (x2 - left) ** 2 + (y2 - bot) ** 2
    d3 = (x3 - left) ** 2 + (y3 - bot) ** 2
    d4 = (x4 - left) ** 2 + (y4 - bot) ** 2
    _4 = np.argmin([d1, d2, d3, d4])

    x1, y1 = coord[_1]['x'], coord[_1]['y']
    x2, y2 = coord[_2]['x'], coord[_2]['y']
    x3, y3 = coord[_3]['x'], coord[_3]['y']
    x4, y4 = coord[_4]['x'], coord[_4]['y']
    loc = []
    loc.append({'x': x1, 'y': y1})
    loc.append({'x': x2, 'y': y2})
    loc.append({'x': x3, 'y': y3})
    loc.append({'x': x4, 'y': y4})
    return loc


def json_to_rects_from_str(anns):
    rects = []
    for ann in anns:
        loc = ann['licensePlateLocation']
        loc = correct(loc)
        rect = dict()
        rect['x1'], rect['x2'], rect['x3'], rect['x4'] = loc[0]['x'], loc[1]['x'], loc[2]['x'], loc[3]['x']
        rect['y1'], rect['y2'], rect['y3'], rect['y4'] = loc[0]['y'], loc[1]['y'], loc[2]['y'], loc[3]['y']
        if ('licensePlateText' not in ann) or (
                not ann['licensePlateText'][:-4]
        ):
            rect['readable'] = 0.0
            rect['licensePlateText'] = ''
        else:
            rect['readable'] = 1.0
            rect['licensePlateText'] = ann['licensePlateText']
        # rect['licensePlateState'] = ann['licensePlateState']

        rects.append(rect)
    return rects


def json_to_rects(path):
    with open(path) as f:
        anns = json.load(f)
        return json_to_rects_from_str(anns)
