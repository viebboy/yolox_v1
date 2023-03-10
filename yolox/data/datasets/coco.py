#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from loguru import logger
import cv2
import numpy as np
from pycocotools.coco import COCO
import tempfile
import random
import string

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset
import mxnet
import joblib
import dill
import copy


def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)


def write_cache(dataset, cache_file, index_file, start_idx, stop_idx):
    record = mxnet.recordio.MXIndexedRecordIO(index_file, cache_file, 'w')
    for i, idx in enumerate(range(start_idx, stop_idx)):
        sample = dataset.get_raw_item(idx)
        byte_rep = dill.dumps(sample)
        record.write_idx(i, byte_rep)
    record.close()
    print(f'complete writing cache file: {cache_file}')


class COCODataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        nb_shard=32,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        self.use_cache = cache
        super().__init__(img_size)

        if not cache:
            logger.info('caching disabled')
            self.prepare(data_dir, json_file, name, img_size, preproc)
            self.use_cache = False
        else:
            logger.info('caching enabled')
            self.preproc = preproc
            self.use_cache = True
            self.nb_shard = nb_shard
            self.records = []
            require_write = self.get_cache_files(data_dir, nb_shard)
            if require_write:
                logger.info('need to write cache')
                metadata = self.prepare(data_dir, json_file, name, img_size, preproc)
                self.write_cache(metadata)
            self.load_cache()

    def get_cache_files(self, data_dir, nb_shard):
        self.cache_files = []
        self.index_files = []
        self.metadata_file = os.path.join(data_dir, 'data_shard.metadata')

        require_write = False

        prefix = os.path.join(data_dir, 'data_shard')
        for i in range(nb_shard):
            f = prefix + '_{:04d}.bin'.format(i)
            self.cache_files.append(f)
            self.index_files.append(f.replace('.bin', '.idx'))

        for f in self.cache_files + self.index_files:
            if not os.path.exists(f):
                require_write = True
                break

        if not os.path.exists(self.metadata_file):
            require_write = True

        return require_write

    def write_cache(self, metadata):
        start_indices = []
        stop_indices = []
        shard_size = int(np.ceil(len(self) / self.nb_shard))
        for i in range(self.nb_shard):
            start_indices.append(i*shard_size)
            stop_indices.append(min((i+1)*shard_size, len(self)))

        joblib.Parallel(n_jobs=-1, backend='loky')(
            joblib.delayed(write_cache)(
                copy.deepcopy(self),
                self.cache_files[i],
                self.index_files[i],
                start_indices[i],
                stop_indices[i]
            )
            for i in range(self.nb_shard)
        )
        self.shard_size = shard_size
        logger.info('complete writing all cache files')

        with open(self.metadata_file, 'wb') as fid:
            dill.dump(metadata, fid, recurse=True)

    def load_cache(self):
        self.records = []
        self.total_samples = 0
        for cache_file, index_file in zip(self.cache_files, self.index_files):
            self.records.append(
                mxnet.recordio.MXIndexedRecordIO(index_file, cache_file, 'r')
            )
            with open(index_file, 'r') as fid:
                content = fid.read().split('\n')[:-1]
                self.total_samples += len(content)
        self.shard_size = int(np.ceil(self.total_samples / self.nb_shard))

        with open(self.metadata_file, 'rb') as fid:
            metadata = dill.load(fid)
            for key, value in metadata.items():
                setattr(self, key, value)
            tmp_anno_file = os.path.join(
                tempfile.gettempdir(),
                ''.join([random.choice(string.ascii_letters) for _ in range(32)]) + '.json'
            )
            with open(tmp_anno_file, 'w') as fid:
                fid.write(metadata['coco_content'])

            self.coco = COCO(tmp_anno_file)

    def prepare(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(416, 416),
        preproc=None
    ):
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file

        anno_file = os.path.join(self.data_dir, "annotations", self.json_file)
        self.coco = COCO(anno_file)
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.imgs = None
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()
        self.total_samples = len(self.ids)

        with open(anno_file, 'r') as fid:
            coco_content = fid.read()

        metadata = {
            'class_ids': self.class_ids,
            'cats': self.cats,
            '_classes': self._classes,
            'annotations': self.annotations,
            'coco_content': coco_content,
        }
        return metadata

    def __len__(self):
        return self.total_samples

    def __del__(self):
        if self.use_cache:
            for record in self.records:
                record.close()
        else:
            del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = os.path.join(self.data_dir, f"img_resized_cache_{self.name}.array")
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for COCO"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    def pull_item(self, index):
        index = int(index)
        if self.use_cache:
            shard_idx = int(np.floor(index / self.shard_size))
            idx_in_shard = index - self.shard_size * shard_idx
            byte_rep = self.records[shard_idx].read_idx(idx_in_shard)
            img, target, img_info, img_id = dill.loads(byte_rep)
            return img, target, img_info, img_id
        else:
            id_ = self.ids[index]

            res, img_info, resized_info, _ = self.annotations[index]
            if self.imgs is not None:
                pad_img = self.imgs[index]
                img = pad_img[: resized_info[0], : resized_info[1], :].copy()
            else:
                img = self.load_resized_img(index)

            return img, res.copy(), img_info, np.array([id_])

    def get_raw_item(self, index):
        id_ = self.ids[index]

        res, img_info, resized_info, _ = self.annotations[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        return img, res.copy(), img_info, np.array([id_])

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        if self.use_cache:
            shard_idx = int(np.floor(index / self.shard_size))
            idx_in_shard = index - self.shard_size * shard_idx
            byte_rep = self.records[shard_idx].read_idx(idx_in_shard)
            img, target, img_info, img_id = dill.loads(byte_rep)
        else:
            img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
