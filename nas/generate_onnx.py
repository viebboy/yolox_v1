from cvinfer.common import load_module
import argparse
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser("generate onnx files from exp files")
    parser.add_argument("--exp-dir", required=True, type=str, help="path to load exp files")
    parser.add_argument("--onnx-dir", required=True, type=str, help="path to save onnx files")
    parser.add_argument("--batch-size", default=None, type=int, help="batch size")
    parser.add_argument("--op-set", default=11, type=int, help="op set")

    return parser.parse_args()


def main():
    args = parse_args()
    exp_files = []
    for f in os.listdir(args.exp_dir):
        if f.endswith(".py"):
            exp_files.append(os.path.join(args.exp_dir, f))

    for idx, f in enumerate(exp_files):
        exp_constructor = load_module(f, 'Exp', 'exp_constructor' + str(idx))
        exp = exp_constructor()
        model = exp.get_deploy_model()
        if args.batch_size is not None:
            dummy_input = torch.rand(args.batch_size, 3, 384, 384)
            dynamic_axes = None
        else:
            dummy_input = torch.rand(1, 3, 384, 384)
            dynamic_axes = {
                'input': {0: 'batch'},
                'output': {0: 'batch'}
            }
        f_out = os.path.join(args.onnx_dir, os.path.basename(f).replace(".py", ".onnx"))

        torch.onnx.export(
            model,
            dummy_input,
            f_out,
            input_names=['input',],
            output_names=['output',],
            dynamic_axes=dynamic_axes,
            opset_version=args.op_set,
            do_constant_folding=True,
        )


if __name__ == '__main__':
    main()
