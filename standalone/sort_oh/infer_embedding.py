import argparse
import dill
import onnxruntime
import os
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=str, help="input directory")
    parser.add_argument("--output-path", required=True, type=str, help="output file path")
    parser.add_argument("--onnx-path", required=True, type=str, help="path to load onnx model")

    return parser.parse_args()

class EmbeddingModel:
    def __init__(self, onnx_path):
		# load preprocessing function
        preprocess_file = onnx_path.replace('.onnx', '.preprocess')

        with open(preprocess_file, 'rb') as fid:
            self.preprocess_function = dill.load(fid)

        #load onnx model from onnx_path
        self.onnx_model =  onnxruntime.InferenceSession(onnx_path)
		# similarly, load postprocessing function
        postprocess_file = onnx_path.replace('.onnx', '.postprocess')
        with open(postprocess_file, 'rb') as fid:
            self.postprocess_function = dill.load(fid)

        self.input_name = self.onnx_model.get_inputs()[0].name

    def __call__(self, x):
        x = self.preprocess_function(x)
        # compute ONNX Runtime output prediction
        ort_inputs = {self.input_name: x}
        x = self.onnx_model.run(None, ort_inputs)[0]
        x = self.postprocess_function(x)

        return x


# loop through input directory, load image, make prediction
def generate_results(input_dir, model):

    results = {}

    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        results[input_path] = model(input_path)

    return results


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_path = args.output_path
    onnx_path = args.onnx_path

    model = EmbeddingModel(onnx_path)

    results = generate_results(input_dir, model)

    with open(output_path, 'wb') as f:
        dill.dump(results, f)

    print('complete inference')

if __name__ == '__main__':
    main()
