# type: ignore
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:53:05 2019

@author: hnguyen

From https://git.taservs.net/axon-research/Plate-Detection-Trainer/blob/d7e92e467b85d15dc57adb1b41bdcb081d81b5d0/utils/augmentations.py
"""
import math
import os
import random

import cv2
import imutils
import numpy as np
import pylab as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import crop
from omegaconf import OmegaConf

from .edge_regression import rotate_along_axis

OUT_IMG_DIR = '../Data/tmp/'
CFG = OmegaConf.load('outputs/configs.yaml')

def show(
        im: np.array,
        ttl: str = '',
        save_dir: str = None,
        bgr: int = 0,
) -> None:
    if bgr:
        im = im[:, :, ::-1]
    plt.imshow(im)
    plt.title(ttl)
    if save_dir is not None:
        plt.savefig(save_dir)
    else:
        plt.show()
    return


def horizontal_flip_darknet(images, targets):
    images = np.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def horizontal_flip_tlbr(image, boxes):
    _, width, _ = image.shape
    image = image[:, ::-1]

    boxes_new = []
    for box in boxes:
        box_new = {}
        box_new['x1'] = width - box['x2']
        box_new['x2'] = width - box['x1']
        for key in box:
            if key not in box_new:
                box_new[key] = box[key]
        boxes_new.append(box_new)

    return image, boxes_new


def bb_convert(x1, y1, x2, y2, x3, y3, x4, y4):
    left = min([x1, x2, x3, x4])
    right = max([x1, x2, x3, x4])
    top = min([y1, y2, y3, y4])
    bot = max([y1, y2, y3, y4])

    return left, right, top, bot


def rotateImage(image, angle):
    row, col, chan = image.shape
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
    new_image = cv2.warpAffine(image.copy(), rot_mat, (col, row))
    return new_image


def image_normalize(img, rects):
    rows, cols, ch = img.shape
    w_normal = 800
    h_normal = 450
    BLACK = [0, 0, 0]
    if rows / cols > h_normal / w_normal:
        height = img.shape[0]
        width = int(np.round(height * w_normal / h_normal))
        pad = abs(int(round((img.shape[1] - width) / 2)))

        img = cv2.copyMakeBorder(
            img,
            0,
            0,
            pad,
            pad,
            cv2.BORDER_CONSTANT,
            value=BLACK,
        )
        rescale = float(h_normal) / img.shape[0]

        for i in range(len(rects)):
            rects[i]['x1'] = (rects[i]['x1'] + pad) * rescale
            rects[i]['x2'] = (rects[i]['x2'] + pad) * rescale
            rects[i]['y1'] = rects[i]['y1'] * rescale
            rects[i]['y2'] = rects[i]['y2'] * rescale
    else:
        width = img.shape[1]
        height = int(np.round(width * h_normal / w_normal))
        pad = abs(int(round((img.shape[0] - height) / 2)))

        img = cv2.copyMakeBorder(
            img,
            pad,
            pad,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=BLACK,
        )
        rescale = float(w_normal) / img.shape[1]

        for i in range(len(rects)):
            rects[i]['x1'] = rects[i]['x1'] * rescale
            rects[i]['x2'] = rects[i]['x2'] * rescale
            rects[i]['y1'] = (rects[i]['y1'] + pad) * rescale
            rects[i]['y2'] = (rects[i]['y2'] + pad) * rescale

    img2 = cv2.resize(img, (w_normal, h_normal))

    return img2, rects


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = []
for i in range(3, 9):
    clahe.append(cv2.createCLAHE(clipLimit=3., tileGridSize=(i, i)))


def random_clahe(img):
    # convert from RGB to LAB color space
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img)  # split on 3 different channels

    irand = np.int(np.floor(np.random.rand() * len(clahe)))
    l2 = clahe[irand].apply(l)  # apply CLAHE to the L-channel

    img = cv2.merge((l2, a, b))  # merge channels
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)  # convert from LAB to RGB

    return img


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """

    angle = angle / 180 * math.pi

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.int(np.round(qx)), np.int(np.round(qy))


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype('uint8')

    return cv2.LUT(image, table)


def add_motion_blur(img, size, angle):
    img_ori = img.copy()
    out = cv2.resize(img, (img_ori.shape[1] * 2, img_ori.shape[0] * 2))
    out = imutils.rotate_bound(out, angle)
    #    generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    # applying the kernel to the input image
    out = cv2.filter2D(out, -1, kernel_motion_blur)
    out = imutils.rotate_bound(out, -angle)

    a, b, c = list((np.round((np.array(out.shape) / 2))).astype(int))
    out = out[
        a - img_ori.shape[0] + 2:a + img_ori.shape[0]
        - 2, b - img_ori.shape[1] + 2:b + img_ori.shape[1] - 2
    ]

    return out


def _rotate(img_in, rects, angle):
    # process absolute coors

    height, width, channel = img_in.shape
    center = tuple(np.array([height, width]) / 2)
    for _ in range(10):

        mask = np.ones((len(rects))).astype(bool)
        res_rects = []
        img = rotateImage(img_in.copy(), angle)
        for i in range(len(rects)):

            x1 = rects[i]['x1']
            x2 = rects[i]['x2']
            y1 = rects[i]['y1']
            y2 = rects[i]['y2']

            x3, y3 = rotate(center, (x2, y1), angle)
            x4, y4 = rotate(center, (x1, y2), angle)
            x1, y1 = rotate(center, (x1, y1), angle)
            x2, y2 = rotate(center, (x2, y2), angle)

            x1 = min(x1, x2, x3, x4)
            x2 = max(x1, x2, x3, x4)
            y1 = min(y1, y2, y3, y4)
            y2 = max(y1, y2, y3, y4)

            wi = x2 - x1
            he = y2 - y1

            if x1 > -wi / 2 and x2 - width < wi / 2 and y1 > -he / \
                    2 and y2 - height < he / 2 and wi >= 7 and he >= 7:

                x1 = max(x1, 1)
                x2 = min(x2, width - 1)
                y1 = max(y1, 1)
                y2 = min(y2, height - 1)
                rect = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
                if 'readable' in rects[i]:
                    rect['readable'] = rects[i]['readable']
                res_rects.append(rect)
            else:
                mask[i] = 0

        if len(res_rects) > 0:
            return img, res_rects

    return img_in, rects


def draw_img(name, frame_aug, rects):
    # convert back to bgr for saving
    frame_aug = cv2.cvtColor(frame_aug, cv2.COLOR_RGB2BGR)
    for rect in rects:
        cv2.rectangle(
            frame_aug,
            (int(rect['x1']), int(rect['y1'])),
            (int(rect['x2']), int(rect['y2'])),
            (0, 255, 0), 2,
        )
    # cv2.putText(frame_aug, str(note), (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
    # 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    if name is not None:
        os.makedirs(os.path.dirname(name), exist_ok=True)
        print('Saving {}'.format(name))
        # frame_aug = cv2.cvtColor(frame_aug, cv2.COLOR_RGB2BGR)
        cv2.imwrite(name, frame_aug)
    else:
        plt.imshow(cv2.cvtColor(frame_aug, cv2.COLOR_BGR2RGB))
        plt.show()


def adjust_intensity(
        plate: np.array,
        _intensity_scale: float,
        vis: int = 0,
) -> np.array:
    show(plate) if vis else 1
    y_cr_cb = cv2.cvtColor(plate, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(y_cr_cb)

    y = np.clip(y.astype(np.float32) * _intensity_scale, 0, 255)
    y = y.astype(np.uint8)

    plate = cv2.merge((y, cr, cb))
    plate = cv2.cvtColor(plate, cv2.COLOR_YCR_CB2RGB)
    show(plate) if vis else 1
    return plate


def zoom(image, boxes=None, zoom_range=(1.0, 1.0)):
    crop_from_entire_image = False
    final_img = np.ones_like(image) * 114
    save_img = 0
    if save_img:
        fn = str(np.random.randint(0, 100000))
        draw_img(OUT_IMG_DIR + fn + '_before_zoom.jpg', image.copy(), boxes)

    # rects: num_boxes x 4 -> rects[0] = [xmin, ymin, xmax, ymax]
    rects = torch.tensor([[b['x1'], b['y1'], b['x2'], b['y2']] for b in boxes])

    zoom_factor = torch.FloatTensor(1).uniform_(
        zoom_range[0], zoom_range[1],
    ).item()
    rects *= zoom_factor

    crop_size = image.shape[:2]
    new_height = int(image.shape[0] * zoom_factor)
    new_width = int(image.shape[1] * zoom_factor)

    resize = transforms.Resize((new_height, new_width))
    image = resize(Image.fromarray(image))

    if crop_from_entire_image:
        crop_top = torch.randint(
            0,
            new_height
            - crop_size[0],
            size=(
                1,
            ),
        ).item()
        crop_left = torch.randint(
            0,
            new_width
            - crop_size[1],
            size=(
                1,
            ),
        ).item()
    else:
        rect_ = random.choice(rects)
        w_, h_ = rect_[0], rect_[1]
        crop_left = max(w_ - crop_size[1] // 2, 0)
        crop_top = max(h_ - crop_size[0] // 2, 0)

    image = crop(
        image,
        int(crop_top),
        int(crop_left),
        crop_size[0],
        crop_size[1],
    )
    image = np.array(image)
    crp_h, crp_w = image.shape[:2]
    final_img[:crp_h, :crp_w, :] = image

    rects[:, [0, 2]] -= crop_left
    rects[:, [1, 3]] -= crop_top
    rects[:, [0, 2]] = torch.clamp(rects[:, [0, 2]], min=0, max=crop_size[1])
    rects[:, [1, 3]] = torch.clamp(rects[:, [1, 3]], min=0, max=crop_size[0])

    for i, b in enumerate(boxes):
        b['x1'], b['y1'], b['x2'], b['y2'] = rects[i].tolist()

    if save_img:
        draw_img(OUT_IMG_DIR + fn + '_after_zoom.jpg', image, boxes)

    return image, boxes


def augment(img, rects, output_dir, save_img, preprocess_config):
    if np.random.rand() < 0.2:
        return img, rects

    fer = 'fer' in CFG['defaults/data']

    zoom_range = [1., 1.]
    if 'augment' in CFG.data:
        zoom_range = [float(_z) for _z in \
                      CFG.data.augment.zoom_range.split(',')]

    if zoom_range[0] != 1. or zoom_range[1] != 1.:
        try:
            img, rects = zoom(
                img, rects, zoom_range=zoom_range,
            )
        except BaseException:
            pass

    # show(img) #--> img is RGB
    # Apply plate-based noise + darkening for dark frames.
    dark_frame_augm_prob = .2
    img_l = round(np.mean(cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:, :, 1]))
    applied_plate_darkness = False
    vis = 0
    if img_l < 60 and np.random.rand() < dark_frame_augm_prob and not fer:
        # imid = np.random.randint(0, 10000000000)
        show(img, f'frame before with L={img_l}') if vis else 1
        for rect in rects:
            try:
                _x1, _x2 = int(round(rect['x1'])), int(round(rect['x2']))
                _y1, _y2 = int(round(rect['y1'])), int(round(rect['y2']))
                _plate = img[_y1:_y2, _x1:_x2, :]
                show(_plate, f'plate before L={img_l}') if vis else 1

                # Add Gaussian pixel noise to the plate
                _plate = _plate.astype(float)
                _plate += np.random.normal(
                    scale=np.random.uniform(
                        0, 0.05,
                    ), size=_plate.shape,
                ) * 255
                _plate = np.array(np.clip(_plate, 0, 255), dtype=np.uint8)

                # Darken the plate
                _plate = adjust_intensity(
                    plate=_plate,
                    _intensity_scale=np.random.uniform(.5, .7),
                    vis=0,
                )
                show(_plate, f'plate after L={img_l}') if vis else 1
                img[_y1:_y2, _x1:_x2, :] = _plate
                applied_plate_darkness = True

            except BaseException:
                pass
        show(img, f'frame after') if vis else 1

    # Apply plate-based adjust_gamma or adjust_intensity.
    plate_augm_prob = .2
    if not applied_plate_darkness and not fer \
            and np.random.rand() < plate_augm_prob:
        plate_gamma = 0.5
        # plate_gamma_max < bright < 1
        bright = plate_gamma + np.random.rand() * (1.0 - plate_gamma)

        # make it brighter with prob=.3
        bright = max(
            1.0 / bright,
            1.5,
        ) if np.random.rand() < 0.3 else bright

        # plate-wise data augm:
        for rect in rects:
            try:
                _x1, _x2 = int(round(rect['x1'])), int(round(rect['x2']))
                _y1, _y2 = int(round(rect['y1'])), int(round(rect['y2']))
                # Apply either adjust_gamma or adjust_intensity data augm.
                if np.random.rand() < 0.5:
                    img[_y1:_y2, _x1:_x2, :] = adjust_gamma(
                        img[_y1:_y2, _x1:_x2, :],
                        gamma=bright,
                    )
                else:
                    img[_y1:_y2, _x1:_x2, :] = adjust_intensity(
                        img[_y1:_y2, _x1:_x2, :],
                        _intensity_scale=np.random.uniform(0.6, 1.1),
                        vis=0,
                    )
            except BaseException:
                pass

    # Flip
    if np.random.rand() > 0.5:
        img, rects = horizontal_flip_tlbr(img, rects)

    # Rotate
    if np.random.rand() > 0.25:
        angle = int(preprocess_config['rotate_max']) * np.random.rand()
        if np.random.rand() > 0.5:
            angle *= -1
        img, rects = _rotate(img, rects, angle=angle)

    # Perspective Transform
    try:
        ratio_ok = min([
            (rect['x2'] - rect['x1'])
            / (rect['y2'] - rect['y1']) for rect in rects
        ]) > 1.5
        height_ok = min([
            (rect['y2'] - rect['y1'])
            / img.shape[1] * 1920 for rect in rects
        ]) > 18
    except BaseException:
        height_ok, ratio_ok = 0, 0

    if np.random.rand() > 0.5 and ratio_ok and height_ok:
        x_rotation = int(
            preprocess_config['rotate_max'],
        ) * np.random.rand()
        y_rotation = 75 * (np.random.rand() * 0.5 + 0.5)
    else:
        x_rotation = 0
        y_rotation = 0

    # Apply a more severe X-Y rotation for FER
    if fer:
        x_rotation *= 1.1
        y_rotation *= 1.1

    z_rotation = 0
    # Apply Z rotation for FER
    # if np.random.rand() > 0.5 and fer:
    #     z_rotation = int(preprocess_config['rotate_max']) * np.random.rand()

    save_rotation_res = 0
    if save_rotation_res:
        fn = str(np.random.randint(0, 100000))
        draw_img(OUT_IMG_DIR + fn + '_before_rot.jpg', img, rects)

    if x_rotation + y_rotation + z_rotation > 0:
        if np.random.rand() > 0.5:
            x_rotation *= -1
        if np.random.rand() > 0.5:
            y_rotation *= -1
        # if np.random.rand() > 0.5: z_rotation *= -1

        points_np = np.array(
            [
                [
                    (rect['x1'], rect['y1']),
                    (rect['x2'], rect['y1']),
                    (rect['x2'], rect['y2']),
                    (rect['x1'], rect['y2']),
                ] for rect in rects
            ], dtype=np.float32,
        )

        img, boxes = rotate_along_axis(
            img, x_rotation, y_rotation, z_rotation, points_np,
        )
        # print(img.shape)

        rects_new = []
        for box, rect in zip(boxes, rects):
            rect_new = {}
            rect_new['x1'] = min(box[:, 0])
            rect_new['y1'] = min(box[:, 1])
            rect_new['x2'] = max(box[:, 0])
            rect_new['y2'] = max(box[:, 1])
            rect_new['readable'] = rect['readable']

            rects_new.append(rect_new)

        rects = rects_new

    if save_rotation_res:
        draw_img(OUT_IMG_DIR + fn + '_after_rot.jpg', img, rects)

    # random clahe:
    if np.random.rand() > 0.8:
        img = random_clahe(img)

    # Add normal (Gaussian) pixel noise
    if np.random.rand() < 0.25:
        g_scale = 0.05
        img = img + np.random.normal(scale=g_scale, size=img.shape) * 255
        img = np.clip(img, 0, 255)
        img = np.array(img, dtype=np.uint8)

    # Gamma adjustment
    if np.random.rand() > 0.25:
        bright = (
            float(preprocess_config['gamma_max'])
            + np.random.rand()
            * (1.0 - float(preprocess_config['gamma_max']))
        )
        if np.random.rand() < 0.5:
            bright = 1.0 / bright
        img = adjust_gamma(img, gamma=bright)

    # Gaussian blur
    if np.random.rand() > 0.75:
        # size = int(round(3 + 2*int(np.random.rand() * 3)))
        size = 0
        while size % 2 == 0:
            size = np.random.randint(
                int(preprocess_config['gauss_min']) * img.shape[0] / 512, int(
                    preprocess_config['gauss_max'],
                ) * img.shape[0] / 512,
            )
        img = cv2.GaussianBlur(img, (size, size), 0)

    if save_img:
        draw_img(
            os.path.join(
                output_dir,
                'random/',
                str(
                    np.random.randint(
                        0,
                        100000,
                    ),
                ) + '_random.jpg',
            ),
            img.copy(),
            rects,
            note='',
        )
    # show(img, 'after')  # --> img is RGB

    return img, rects
