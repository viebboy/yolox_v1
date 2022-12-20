#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger
import json

import torch
from torch import nn
import numpy as np

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module

import onnxruntime as RT

def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="input", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

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

def verify_onnx_model(batch_shape, pytorch_model, onnx_model):
    for _ in range(10):
        inputs = np.random.rand(*batch_shape).astype('float32')
        onnx_out = onnx_model(inputs).flatten()
        with torch.no_grad():
            inputs = torch.from_numpy(inputs)
            torch_out = pytorch_model(inputs).numpy().flatten()
        if not np.allclose(onnx_out, torch_out, rtol=1e-3, atol=1e-3):
            print('mismatched outputs')
            print('onnx outputs')
            print(onnx_out[:10])
            print('torch outputs')
            print(torch_out[:10])
            raise RuntimeError()

    logger.info('complete verifying onnx model')

def verify_pytorch_model(spatial_dim, old_model, new_model):
    old_model.eval()
    new_model.eval()

    with torch.no_grad():
        for _ in range(10):
            x = torch.randn(1, 3, spatial_dim[0], spatial_dim[1]).float()
            y1 = old_model(x).numpy().flatten()
            y2 = new_model(x).numpy().flatten()
            if not np.allclose(y1, y2, rtol=1e-3, atol=1e-3):
                print('mismatched outputs')
                print(f'y1, shape: {y1.shape}')
                print(y1.flatten()[:10])
                print(f'y2, shape: {y2.shape}')
                print(y2.flatten()[:10])
                raise RuntimeError()


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    assert args.output_name.endswith('.onnx'), 'Output onnx file must end with .onnx'
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    deploy_model = exp.get_deploy_model()

    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    if "model" in ckpt:
        ckpt = ckpt["model"]

    model.load_state_dict(ckpt)
    deploy_model.load_weights_from(model)

    deploy_model = replace_module(deploy_model, nn.SiLU, SiLU)

    model.float()
    deploy_model.float()
    model.eval()
    deploy_model.eval()

    verify_pytorch_model(exp.test_size, model, deploy_model)

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])
    with torch.no_grad():
        outputs = deploy_model(dummy_input)

    torch.onnx._export(
        deploy_model,
        dummy_input,
        args.output_name,
        input_names=[args.input],
        output_names=[args.output],
        dynamic_axes={args.input: {0: 'batch'},
                      args.output: {0: 'batch'}} if args.dynamic else None,
        opset_version=args.opset,
    )
    logger.info("generated onnx model named {}".format(args.output_name))
    metadata_file = args.output_name.replace('.onnx', '.json')
    metadata = {'input_size': exp.test_size, 'batch_size': args.batch_size, 'class_names': exp.get_class_names()}
    with open(metadata_file, 'w') as fid:
        fid.write(json.dumps(metadata, indent=2))
    logger.info("generated metadata file at {}".format(metadata_file))

    onnx_model = OnnxModel(args.output_name)
    verify_onnx_model(
        (args.batch_size, 3, exp.test_size[0], exp.test_size[1]),
        deploy_model,
        onnx_model
    )

    if not args.no_onnxsim:
        import onnx

        from onnxsim import simplify

        input_shapes = {args.input: list(dummy_input.shape)} if args.dynamic else None

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        logger.info('simplifying onnx model')
        model_simp, check = simplify(onnx_model,
                                     dynamic_input_shape=args.dynamic,
                                     input_shapes=input_shapes)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))

        onnx_model = OnnxModel(args.output_name)
        verify_onnx_model(
            (args.batch_size, 3, exp.test_size[0], exp.test_size[1]),
            deploy_model,
            onnx_model
        )


if __name__ == "__main__":
    main()
