from cvinfer.common import load_module
import argparse
import os
import torch
import json
from model_factory_v5 import Exp

def parse_args():
    parser = argparse.ArgumentParser("generate onnx model factory")
    parser.add_argument("--onnx-dir", required=True, type=str, help="path to save onnx files")
    parser.add_argument("--config", required=True, type=str, help="path to json config file")

    return parser.parse_args()


def main():
    args = parse_args()
    config = json.load(open(args.config))
    exp = Exp()
    exp.create_factory(
        space_config=config['model_space'],
        output_path=args.onnx_dir,
        batch_size=config['batch_size'],
        opset_version=config['opset_version'],
        do_constant_folding=config['do_constant_folding'],
    )

if __name__ == '__main__':
    main()
