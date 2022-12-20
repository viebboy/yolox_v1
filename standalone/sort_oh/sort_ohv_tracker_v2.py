#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger
import cv2
import torch
import torchvision
import numpy as np
import json
import onnxruntime as RT
from sort_ohv import SORT_OH
import threading
from queue import Queue


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def parse_args():
    parser = argparse.ArgumentParser("YOLOX ONNX inference!")
    parser.add_argument("--tracker-config", required=True, type=str, help="path to json tracker configuration")
    parser.add_argument(
        "--input-type", required=True, choices=['image', 'video', 'webcam', 'rtsp'])
    parser.add_argument(
        "--skip-frame", default='False', choices=['true', 'True', 'false', 'False'])
    parser.add_argument(
        "--wide-angle", default='True', choices=['true', 'True', 'false', 'False'])
    parser.add_argument(
        "--show-tracking", default='True', choices=['true', 'True', 'false', 'False'])
    parser.add_argument(
        "--draw-motion-estimation", default='False', choices=['true', 'True', 'false', 'False'])
    parser.add_argument("--onnx-file", type=str, required=True)
    parser.add_argument(
        "--input-path", type=str, default=None, help="path to images or video or RTSP path"
    )
    parser.add_argument(
        "--output-path", type=str, default=None, help="path to save the resulting images or video"
    )

    parser.add_argument(
        "--execution-provider",
        default="CPUExecutionProvider",
        choices=['CPUExecutionProvider'],
        type=str,
        help="execution provider to use",
    )
    parser.add_argument("--confidence-threshold", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms-threshold", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--output-fps", default=5, type=int, help="output fps for visualization")
    parser.add_argument("--fuzzy-width", default=0.1, type=float, help="fuzzy width boundary")
    parser.add_argument("--fuzzy-height", default=0.1, type=float, help="fuzzy height boundary")
    parser.add_argument("--center-crop-ratio", default=0.6, type=float,
                        help="crop ratio from the center point")

    args = parser.parse_args()
    if args.skip_frame in ['true', 'True']:
        args.skip_frame = True
    else:
        args.skip_frame = False

    if args.draw_motion_estimation in ['true', 'True']:
        args.draw_motion_estimation = True
    else:
        args.draw_motion_estimation = False

    if args.show_tracking in ['true', 'True']:
        args.show_tracking = True
    else:
        args.show_tracking = False

    if args.wide_angle in ['true', 'True']:
        args.wide_angle = True
    else:
        args.wide_angle = False

    return args

class OnnxModel:
    def __init__(self, engine_file, provider='CPUExecutionProvider'):
        print('all available execution providers')
        print('---')
        providers = RT.get_available_providers()
        for p in providers:
            print(p)
        print('---')
        print(f'trying to run with {provider}')
        print('---')
        self.session = RT.InferenceSession(
            engine_file,
            providers=[provider]
        )

    def __call__(self, inputs: np.ndarray):
        output = self.session.run([], {'input': inputs})[0]
        return output

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def visualize_annotation(
    img,
    boxes,
    scores,
    cls_ids,
    target_ids,
    predicted_positions,
    conf=0.5,
    class_names=None,
    crop_box=None,
    fuzzy_box=None,
):
    if crop_box is not None:
        cv2.rectangle(
            img,
            (crop_box[0], crop_box[1]), (crop_box[2], crop_box[3]),
            (255, 255, 255),
            4
        )

    if fuzzy_box is not None:
        cv2.rectangle(
            img,
            (fuzzy_box[0], fuzzy_box[1]), (fuzzy_box[2], fuzzy_box[3]),
            (20, 20, 20),
            4
        )

    if boxes is None:
        boxes = []
    else:
        scores = scores.flatten()

    for i in range(len(boxes)):
        box = boxes[i]
        target_id = target_ids[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (COLORS[cls_id] * 255).astype(np.uint8).tolist()
        if target_id is None:
            color = (255, 0, 0)
        else:
            nb_color = len(COLORS)
            color_idx = int(target_id) - np.floor(int(target_id) / nb_color)
            color = (COLORS[int(color_idx)] * 255).astype(np.uint8).tolist()

        if target_id is not None:
            text = '{} id={}'.format(class_names[cls_id], target_id)
        else:
            text = '{} conf={} %'.format(class_names[cls_id], int(score*100))

        txt_color = (0, 0, 0) if np.mean(COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 4)

        txt_bk_color = (COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(2.0*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)


    for i in range(len(predicted_positions)):
        box = predicted_positions[i]
        target_id = box[0]
        x0 = int(box[1])
        y0 = int(box[2])
        x1 = int(box[3])
        y1 = int(box[4])

        color = (0, 0, 0)

        text = 'person id={}'.format(target_id)
        txt_color = (0, 0, 0) if np.mean(COLORS[0]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 5)

        txt_bk_color = (COLORS[0] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(2.0*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def postprocess(
    prediction,
    num_classes,
    conf_thre=0.7,
    nms_thre=0.45,
    class_agnostic=False
):
    prediction = torch.from_numpy(prediction)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

def preproc(img, input_size, swap=(2, 0, 1)):
    # swap is for BGR to RGB
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class Predictor(object):
    def __init__(
        self,
        input_size,
        model,
        class_names,
        confidence_threshold,
        nms_threshold,
        wide_angle,
        crop_ratio,
        fuzzy_width,
        fuzzy_height,
    ):
        self.model = model
        self.cls_names = class_names
        self.num_classes = len(class_names)
        self.conf_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.wide_angle = wide_angle
        self.crop_ratio = crop_ratio
        self.fuzzy_width = fuzzy_width
        self.fuzzy_height = fuzzy_height
        self.crop_box = None
        self.fuzzy_box = None

    def infer(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        # if wide angle, take a crop
        if self.wide_angle:
            height_pad = int(height * (1 - self.crop_ratio)/2)
            width_pad = int(width * (1 - self.crop_ratio)/2)
            crop_img = img[height_pad:height-height_pad, width_pad:width-width_pad, :]
            crop_img_height = crop_img.shape[0]
            crop_img_width = crop_img.shape[1]
        else:
            height_pad = None
            width_pad = None
            crop_img = None
            crop_img_height = None
            crop_img_width = None

        img_info['height_pad'] = height_pad
        img_info['width_pad'] = width_pad

        img, ratio = preproc(img, self.input_size)
        if crop_img is not None:
            crop_img, crop_img_ratio = preproc(crop_img, self.input_size)
        else:
            crop_img_ratio = None

        img_info["ratio"] = ratio
        img_info["crop_img_ratio"] = crop_img_ratio

        img = np.expand_dims(img, 0)
        if crop_img is not None:
            crop_img = np.expand_dims(crop_img, 0)
            img = np.concatenate((img, crop_img), 0)

        outputs = self.model(img)
        outputs = postprocess(
            outputs,
            self.num_classes,
            self.conf_threshold,
            self.nms_threshold,
            class_agnostic=True
        )

        # process boxes from the outer image
        boxes, confidences, class_ids = self.postprocess(
            outputs[0],
            height,
            width,
            ratio,
            True
        )

        # process inner image
        if self.wide_angle:
            inner_boxes, inner_confidences, inner_class_ids = self.postprocess(
                outputs[1],
                crop_img_height,
                crop_img_width,
                crop_img_ratio,
                False,
            )

            boxes, confidences, class_ids = self.combine(
                boxes,
                confidences,
                class_ids,
                inner_boxes,
                inner_confidences,
                inner_class_ids,
                height,
                width,
                height_pad,
                width_pad,
            )

            if self.crop_box is None:
                self.crop_box = (width_pad, height_pad, width-width_pad, height - height_pad)
            if self.fuzzy_box is None:

                self.fuzzy_box = (
                    width_pad + int(crop_img_width * self.fuzzy_width),
                    height_pad + int(crop_img_height * self.fuzzy_height),
                    width - width_pad - int(crop_img_width * self.fuzzy_width),
                    height - height_pad - int(crop_img_height * self.fuzzy_height)
                )

        return boxes, confidences, class_ids, img_info

    def combine(
        self,
        outter_boxes,
        outter_confidences,
        outter_class_ids,
        inner_boxes,
        inner_confidences,
        inner_class_ids,
        height,
        width,
        height_pad,
        width_pad,
    ):
        if outter_boxes is None and inner_boxes is None:
            return None, None, None
        elif outter_boxes is None and inner_boxes is not None:
            # shift the box to outter image coordinates
            inner_boxes[:, 0] = inner_boxes[:, 0] + width_pad
            inner_boxes[:, 1] = inner_boxes[:, 1] + height_pad
            inner_boxes[:, 2] = inner_boxes[:, 2] + width_pad
            inner_boxes[:, 3] = inner_boxes[:, 3] + height_pad
            return inner_boxes, inner_confidences, inner_class_ids
        elif inner_boxes is None:
            return outter_boxes, outter_confidences, outter_class_ids
        else:
            # shift the box to outter image coordinates
            inner_boxes[:, 0] = inner_boxes[:, 0] + width_pad
            inner_boxes[:, 1] = inner_boxes[:, 1] + height_pad
            inner_boxes[:, 2] = inner_boxes[:, 2] + width_pad
            inner_boxes[:, 3] = inner_boxes[:, 3] + height_pad

            # boundary to combine
            # inner boxes selected inside boundary
            # outter boxes selected outside boundary
            crop_width = width - 2 * width_pad
            crop_height = height - 2 * height_pad
            x_min = width_pad + int(crop_width * self.fuzzy_width)
            y_min = height_pad + int(crop_height * self.fuzzy_height)

            x_max = width - x_min
            y_max = height - y_min

            # select boxes inside boundary
            inner_center_x = (inner_boxes[:, 0] + inner_boxes[:, 2]) / 2
            inner_center_y = (inner_boxes[:, 1] + inner_boxes[:, 3]) / 2
            x_mask = (inner_center_x - x_min) * (x_max - inner_center_x) > 0
            y_mask = (inner_center_y - y_min) * (y_max - inner_center_y) > 0
            mask = [i and j for i, j in zip(x_mask.flatten(), y_mask.flatten())]
            inner_boxes  = inner_boxes[mask]
            inner_confidences  = inner_confidences[mask]
            inner_class_ids = inner_class_ids[mask]

            # select boxes outside boundary
            outter_center_x = (outter_boxes[:, 0] + outter_boxes[:, 2]) / 2
            outter_center_y = (outter_boxes[:, 1] + outter_boxes[:, 3]) / 2
            x_mask = (outter_center_x - x_min) * (x_max - outter_center_x) <= 0
            y_mask = (outter_center_y - y_min) * (y_max - outter_center_y) <= 0
            mask = [i or j for i, j in zip(x_mask.flatten(), y_mask.flatten())]
            outter_boxes  = outter_boxes[mask]
            outter_confidences  = outter_confidences[mask]
            outter_class_ids = outter_class_ids[mask]

            boxes = np.concatenate([outter_boxes, inner_boxes], axis=0)
            confidences = np.concatenate([outter_confidences, inner_confidences], axis=0)
            class_ids = np.concatenate([outter_class_ids, inner_class_ids], axis=0)
            return boxes, confidences, class_ids


    def postprocess(self, output, height, width, ratio, clipping):
        if output is not None and output.numpy().shape[0] > 0:
            output = output.numpy()
            bboxes = output[:, :4]
            bboxes /= ratio
            if clipping:
                bboxes[:, 0] = np.clip(bboxes[:, 0], 0, width - 1)
                bboxes[:, 1] = np.clip(bboxes[:, 1], 0, height - 1)
                bboxes[:, 2] = np.clip(bboxes[:, 2], 0, width - 1)
                bboxes[:, 3] = np.clip(bboxes[:, 3], 0, height - 1)
            confidences = output[:, 4:5] * output[:, 5:6]
            class_ids = output[:, 6]
            #detections = np.concatenate([bboxes, confidences], axis=1)
        else:
            bboxes = None
            confidences = None
            class_ids = None
        return bboxes, confidences, class_ids

    def visualize(
        self,
        bboxes,
        confidences,
        class_indices,
        target_ids,
        predicted_positions,
        img_info,
        cls_conf=0.35
    ):

        ratio = img_info["ratio"]
        img = img_info["raw_img"]

        vis_res = visualize_annotation(
            img,
            bboxes,
            confidences,
            class_indices,
            target_ids,
            predicted_positions,
            cls_conf,
            self.cls_names,
            self.crop_box,
            self.fuzzy_box,
        )
        return vis_res


def image_demo(predictor, input_dir, output_path):
    if os.path.isdir(input_path):
        files = get_image_list(input_path)
    else:
        files = [path]
    files.sort()

    os.makedirs(output_dir, exist_ok=True)

    for image_name in files:
        outputs, img_info = predictor.infer(image_name)
        result_image = predictor.visualize(outputs[0], img_info, predictor.conf_threshold)
        save_file_name = os.path.join(output_dir, os.path.basename(image_name))
        logger.info("Saving detection result in {}".format(save_file_name))
        cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

def capture_from_rtsp(path, image_queue, event_queue, skip_frame):
    cap = cv2.VideoCapture(path)
    count = 0
    start_time = time.time()
    skip_flag = False
    while True:
        ret_val, frame = cap.read()
        count += 1
        if count == 100:
            fps = int(count / (time.time() - start_time))
            logger.info(f'capture FPS: {fps}')
            start_time = time.time()
            count = 0
        if ret_val:
            if skip_frame:
                if not skip_flag:
                    image_queue.put((ret_val, frame))
                    skip_flag = True
                else:
                    skip_flag = False
            else:
                image_queue.put((ret_val, frame))
        else:
            break
        if not event_queue.empty():
            logger.info('receive termination signal')
            break

    cap.release()
    logger.info('RTSP closed')

def imageflow_demo(
    predictor,
    input_path,
    output_path,
    output_fps,
    tracker_kwargs,
    skip_frame,
    draw_motion_estimation,
    show_tracking
):
    cap = cv2.VideoCapture(input_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = SORT_OH(
        img_height=height,
        img_width=width,
        kwargs=tracker_kwargs
    )

    if isinstance(input_path, str) and input_path.startswith('rtsp'):
        cap.release()
        time.sleep(1)
        image_queue = Queue()
        event_queue = Queue()
        cap_thread = threading.Thread(target=capture_from_rtsp,
                                      args=(input_path, image_queue, event_queue, skip_frame))
        cap_thread.start()
        cap = None
    else:
        image_queue = None
        cap_thread = None

    if output_path is not None:
        logger.info(f"video save_path is {output_path}")
        vid_writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (int(width), int(height))
        )

    count = 0
    start_time = time.time()
    is_terminated = False
    while True:
        if cap is not None:
            ret_val, frame = cap.read()
        else:
            while True:
                if not image_queue.empty():
                    ret_val, frame = image_queue.get()
                    break
                else:
                    time.sleep(0.001)

        if ret_val:
            bboxes, confidences, class_ids, img_info = predictor.infer(frame)
            if bboxes is not None:
                detections = np.concatenate([bboxes, confidences], axis=1)
            else:
                detections = None

            if show_tracking:
                target_ids, predicted_positions = tracker.update(detections)
            else:
                target_ids = [None,] * detections.shape[0] if detections is not None else None
                predicted_positions = []

            if not draw_motion_estimation:
                predicted_positions = []
            result_frame = predictor.visualize(
                bboxes,
                confidences,
                class_ids,
                target_ids,
                predicted_positions,
                img_info,
                predictor.conf_threshold,
            )
            if output_path is not None:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("Human Detection & Tracking v1", cv2.WINDOW_NORMAL)
                cv2.imshow("Human Detection & Tracking v1", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                logger.info('exiting...')
                if cap is None:
                    logger.info('closing the thread...')
                    event_queue.put(0)
                    cap_thread.join()
                    is_terminated = True
                break
            count += 1
            if count == 100:
                fps = int(count / (time.time() - start_time))
                count = 0
                start_time = time.time()
                logger.info(f'pipeline FPS: {fps}')
        else:
            break

    logger.info('outside the main loop...')
    if cap_thread is not None and not is_terminated:
        logger.info('closing the thread...')
        event_queue.put(0)
        cap_thread.join()


def load_model(onnx_file, execution_provider):
    assert os.path.exists(onnx_file)
    assert onnx_file.endswith('.onnx'), 'ONNX model file must end with .onnx'
    metadata_file = onnx_file.replace('.onnx', '.json')
    assert os.path.exists(metadata_file), f'metadata file: {metadata_file} doesnt exist'

    with open(metadata_file, 'r') as fid:
        metadata = json.loads(fid.read())

    model = OnnxModel(onnx_file, execution_provider)
    return model, metadata


if __name__ == "__main__":
    args = parse_args()

    # parse tracker params
    assert os.path.exists(args.tracker_config)
    with open(args.tracker_config, 'r') as fid:
        tracker_params = json.loads(fid.read())

    model, metadata = load_model(args.onnx_file, args.execution_provider)
    predictor = Predictor(
        input_size=metadata['input_size'],
        model=model,
        class_names=metadata['class_names'],
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
        wide_angle=args.wide_angle,
        crop_ratio=args.center_crop_ratio,
        fuzzy_width=args.fuzzy_width,
        fuzzy_height=args.fuzzy_height,
    )

    if args.input_type == 'image':
        assert args.output_path is not None, '--output-path must not be None for image inference'
        image_demo(predictor, args.input_path, args.output_path)
    elif args.input_type in ['video', 'webcam', 'rtsp']:
        if args.input_type == 'webcam':
            args.input_path = 0
        imageflow_demo(
            predictor,
            args.input_path,
            args.output_path,
            args.output_fps,
            tracker_params,
            args.skip_frame,
            args.draw_motion_estimation,
            args.show_tracking,
        )
