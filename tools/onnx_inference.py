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


def make_parser():
    parser = argparse.ArgumentParser("YOLOX ONNX inference!")
    parser.add_argument(
        "--input-type", required=True, choices=['image', 'video', 'webcam'])
    parser.add_argument("--onnx-file", type=str, required=True)
    parser.add_argument(
        "--input-path", type=str, required=True, help="path to images or video"
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
    return parser

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

def visualize_annotation(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
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
    ):
        self.model = model
        self.cls_names = class_names
        self.num_classes = len(class_names)
        self.conf_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

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

        img, ratio = preproc(img, self.input_size)
        img_info["ratio"] = ratio

        t0 = time.time()
        outputs = self.model(np.expand_dims(img, 0))
        outputs = postprocess(
            outputs,
            self.num_classes,
            self.conf_threshold,
            self.nms_threshold,
            class_agnostic=True
        )

        logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visualize(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img

        bboxes = output[:, 0:4]
        height = img_info['height']
        width = img_info['width']
        bboxes[:, 0] = torch.clip(bboxes[:, 0], 0, None)
        bboxes[:, 1] = torch.clip(bboxes[:, 1], 0, None)
        bboxes[:, 2] = torch.clip(bboxes[:, 2], None, width-1)
        bboxes[:, 3] = torch.clip(bboxes[:, 3], None, height-1)

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = visualize_annotation(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, input_path, output_path):
    if os.path.isdir(input_path):
        files = get_image_list(input_path)
    else:
        files = [input_path]
    files.sort()

    os.makedirs(output_path, exist_ok=True)

    for image_name in files:
        outputs, img_info = predictor.infer(image_name)
        result_image = predictor.visualize(outputs[0], img_info, predictor.conf_threshold)
        save_file_name = os.path.join(output_path, os.path.basename(image_name))
        logger.info("Saving detection result in {}".format(save_file_name))
        cv2.imwrite(save_file_name, result_image)


def imageflow_demo(predictor, input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if output_path is not None:
        logger.info(f"video save_path is {output_path}")
        vid_writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )

    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.infer(frame)
            result_frame = predictor.visualize(outputs[0], img_info, predictor.conf_threshold)
            if output_path is not None:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def load_model(onnx_file, execution_provider):
    assert os.path.exists(onnx_file)
    assert onnx_file.endswith('.onnx'), 'ONNX model file must end with .onnx'
    metadata_file = onnx_file.replace('.onnx', '.json')
    assert os.path.exists(metadata_file), f'metadata file: {metadata_file} doesnt exist'

    with open(metadata_file, 'r') as fid:
        metadata = json.loads(fid.read())

    assert metadata['batch_size'] == 1, 'Only work with a model having batch size of 1'
    model = OnnxModel(onnx_file, execution_provider)
    return model, metadata


if __name__ == "__main__":
    args = make_parser().parse_args()
    model, metadata = load_model(args.onnx_file, args.execution_provider)
    predictor = Predictor(
        input_size=metadata['input_size'],
        model=model,
        class_names=metadata['class_names'],
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold
    )

    if args.input_type == 'image':
        assert args.output_path is not None, '--output-path must not be None for image inference'
        image_demo(predictor, args.input_path, args.output_path)
    elif args.input_type == 'video':
        imageflow_demo(predictor, args.input_path, args.output_path)
    elif args.input_type == 'webcam':
        webcam_id = int(args.input_path)
        imageflow_demo(predictor, webcam_id, args.output_path)
    else:
        raise RuntimeError(f'Unknown input type: {args.input_type}')
