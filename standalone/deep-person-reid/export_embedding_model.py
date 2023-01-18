import argparse
import dill
import torch
import torchvision.transforms as T
from PIL import Image

import torchreid
import numpy as np
import onnxruntime
from infer_embedding import EmbeddingModel


class FeatureExtractor(torchreid.utils.FeatureExtractor):
    def __init__(
        self,
        model_name='',
        model_path='',
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cpu',
        verbose=True
    ):
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            image_size=image_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            pixel_norm=pixel_norm,
            device=device,
            verbose=verbose,
        )
        self.device = device
        self.img_size = image_size

    def export(self, onnx_path):
        #For example, onnx_path='model.onnx'
        #this function produces 3 files: 'model.onnx', 'model.preprocess', 'model.postprocess'

        #----------------------------------save preprocessing function----------------------------------------
        def preprocessing(input):
            if isinstance(input, list):
                images = []
                for element in input:
                    if isinstance(element, str):
                        image = Image.open(element).convert('RGB')
                    elif isinstance(element, np.ndarray):
                        image = self.to_pil(torch.Tensor(element))
                    else:
                        raise TypeError(
                            'Type of each element must belong to [str | numpy.ndarray]'
                        )

                    image = self.preprocess(image)
                    images.append(image)

                images = torch.stack(images, dim=0)

            elif isinstance(input, str):
                image = Image.open(input).convert('RGB')
                image = self.preprocess(image)
                images = image.unsqueeze(0)

            elif isinstance(input, np.ndarray):
                image = self.to_pil(torch.Tensor(input))
                image = self.preprocess(image)
                images = image.unsqueeze(0)
            else:
                raise NotImplementedError

            images = to_numpy(images)

            return images

        #save preprocessing function
        preprocessing_file = onnx_path.replace('.onnx', '.preprocess')
        with open(preprocessing_file, 'wb') as fid:
            dill.dump(preprocessing, fid)

        #----------------------------------save onnx model----------------------------------------
        # Input to the model
        batch_size = 1    # just a random number
        x = torch.randn(batch_size, 3, self.img_size[0], self.img_size[1], requires_grad=True).to(self.device)
        torch_out = self.model(x)

        # Export the model
        torch.onnx.export(self.model,               # model being run
                        x,                         # model input (or a tuple for multiple inputs)
                        onnx_path,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=10,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                        'output' : {0 : 'batch_size'}})

        #----------------------------------save postprocessing function----------------------------------------
        def postprocessing(input):
            return input

        postprocessing_file = onnx_path.replace('.onnx', '.postprocess')
        with open(postprocessing_file, 'wb') as fid:
            dill.dump(postprocessing, fid)

        #----------------------------------verify onnx model----------------------------------------

        # compute ONNX Runtime output prediction
        ort_session = onnxruntime.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)

        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

        # verify the whole pipeline
        onnx_model = EmbeddingModel(onnx_path)
        for i in range(10):
            x = np.random.rand(3, 300, 300)
            onnx_out = onnx_model(x)
            torch_out = self.__call__(x).numpy()
            np.testing.assert_allclose(onnx_out, torch_out, rtol=1e-03, atol=1e-05)

        print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    def __call__(self, input):
        if isinstance(input, list):
            images = []

            for element in input:
                if isinstance(element, str):
                    image = Image.open(element).convert('RGB')

                elif isinstance(element, np.ndarray):
                    image = self.to_pil(torch.Tensor(element))
                else:
                    raise TypeError(
                        'Type of each element must belong to [str | numpy.ndarray]'
                    )

                image = self.preprocess(image)
                images.append(image)

            images = torch.stack(images, dim=0)
            images = images.to(self.device)

        elif isinstance(input, str):
            image = Image.open(input).convert('RGB')
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, np.ndarray):
            image = self.to_pil(torch.Tensor(input))
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)
        else:
            raise NotImplementedError

        with torch.no_grad():
            features = self.model(images)

        return features

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, type=str, help="name of model")
    parser.add_argument("--model-path", required=True, type=str, help="model checkpoint path")
    parser.add_argument("--onnx-path", required=True, type=str, help="path to save onnx model")
    return parser.parse_args()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    args = parse_args()

    extractor = FeatureExtractor(
        model_name=args.model_name,
        model_path=args.model_path,
    )

    extractor.export(args.onnx_path)


if __name__ == '__main__':
    main()
