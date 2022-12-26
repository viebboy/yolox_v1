## Human Detection and Tracking V1

The source code is based on the [yolox framework](https://github.com/Megvii-BaseDetection/YOLOX)

Please refer to the original README in [yolox framework](https://github.com/Megvii-BaseDetection/YOLOX) for a complete description. 

## Installation

Assuming we are in the root directory of this repo
Install the dependencies by:

```bash
pip3 install -r requirements.txt
```

Then install the package by:

```bash
pip3 install -e .
```

### Detection Model Training
The data must be prepared in the COCO format. Almost all experiment hyperparameters are specified in a python file with an Exp class.
The experiment configurations that correspond to the models described in the [report](https://axon.quip.com/t4IHA4Ab2zLT/Taser-Targeting-Human-Detection-Tracking-Q4-2022-White-Paper) are the following:

- [yolox_nano](./exps/open_image_person_detector_yolox_nano.py)
- [yolox_tiny](./exps/open_image_person_detector_yolox_tiny.py)
- [yolox_custom_v2_exp4](./exps/open_image_person_detector_v2_exp4.py)
- [yolox_custom_v2_exp7](./exps/open_image_person_detector_v2_exp7.py)
- [yolox_custom_v5_exp1](./exps/open_image_person_detector_v5_exp1.py)
- [yolox_custom_v5_exp2](./exps/open_image_person_detector_v5_exp2.py)
- [yolox_custom_v5_exp3](./exps/open_image_person_detector_v5_exp3.py)

Here is an example of training command for `yolox_custom_v2_exp4`:

```bash
python -m yolox.tools.train -f ${CONFIG_FILE} -d 4 -b 48 --fp16 -o --data-dir ${DATA_DIR} --train-ann ${TRAIN_ANN} --val-ann ${VAL_ANN}
```

with:

- `-f`: path to the exp config file
- `-d`: the number of gpus
- `-b`: total batch size for all devices
- `--fp16`: whether to use half precision
- `-o`: whether to occupy gpu first for training
- `--data-dir`: path to directory that contain images.
  This directory must contain `train2017`, `val2017`, `annotations` as subdirs. 
  `train2017` should contain training images
  `val2017` should contain validation images
  `annotations` should contain json annotation files 
- `--train-ann`: name (not the path) of the json training annotation file. This file should be under the `annotations` subdir of the `--data-dir`. 
- `--val-ann`: name (not the path) of the json validation annotation file. This file should be under the `annotations` subdir of the `--data-dir`. 

### Detection Inference
The inference script uses ONNX model format so we need to convert the pytorch checkpoint to the ONNX format with a metadata first. 
Conversion command:

```bash
python -m yolox.tools.export_onnx --output-name {ONNX_FILE} --batch-size 1 --exp-file ${CONFIG_FILE}
```

with:

- `--output-name`: path to the onnx output file. This file should end with .onnx
- `--batch-size`: batch dimension
- `--exp-file`: path to experiment config file that was used to train. Note that the experiment config file knows where is the output dir that contains exp artifacts

After the conversion, we will obtain the onnx model file as well as the json metadata file (same name, ends with .json) needed to run the inference tool

We can run inference on images, video or webcam using this command:

```bash
python -m yolox.tools.onnx_inference --onnx-file ${ONNX_FILE} \
                                     --input-type ${INPUT_TYPE} \
                                     --input-path ${INPUT_PATH} \
                                     --output-path ${OUTPUT_PATH} \
                                     --confidence-threshold ${CONF_THRESHOLD} \
                                     --nms-threshold ${NMS_THRESHOLD}
```

with:

- `--onnx-file`: path to the onnx file generated in the above step
- `--input-type`: the type of input. This can be `image`, `video` or `webcam`
- `--input-path`: path to the input data. If webcam, then it is id of camera (normally 0)
- `--output-path`: path to write overlayed detection. If not specify, live video screen is shown
- `--confidence-threshold': threshold of confidence score
- `--nms-threshold`: non-maximum suppression threshold


### SORT OH Tracker 
SORT OH is a multi-object tracker. It is implemented as standalone code under `./standalone/sort_oh`. 
To use this tracker, we need a detection model in ONNX format converted using the above-mentioned conversion tool. 
The command to run this tracker is the following:

```bash
python standalone/sort_oh/sort_ohv_tracker_v2.py \
    --input-type ${INPUT_TYPE} \
    --onnx-file ${ONNX_FILE} \
    --confidence-threshold ${CONF_THRESHOLD} \
    --nms-threshold ${NMS_THRESHOLD} \
    --tracker-config ${TRACKER_CONFIG} \
    --input-path ${INPUT_PATH} \
    --show-tracking ${SHOW_TRACKING} \
    --wide-angle ${WIDE_ANGLE} \
    --center-crop-ratio ${CENTER_CROP_RATIO} \
    --fuzzy-width ${FUZZY_WIDTH} \
    --fuzzy-height ${FUZZY_HEIGHT} \
    --output-path ${OUTPUT_PATH} \
    --output-fps ${OUTPUT_FPS}
``` 

with:

- `--input-type`: the type of input. This can be `video`, `webcam`, `rtsp`
- `--onnx-file`: path to the onnx file generated in the above step
- `--confidence-threshold': threshold of confidence score
- `--nms-threshold': NMS threshold
- `--tracker-config': path to the json config file for tracker. Example can be found in `./standalone/sort_oh/config.json` 
- `--input-path`: path to the input data. If webcam, then it is id of camera (normally 0)
- `--show-tracking`: whether to show the tracking result. If False, only detection is run. 
- `--wide-angle`: whether to use the cropping trick to improve wide-angle detection. If True, the ONNX model must work with a batch size of 2. 
- `--center-crop-ratio`: the ratio to perform center cropping in the cropping trick. 
- `--fuzzy-width`: the amount of fuzzy width boundary in the cropping trick. 
- `--fuzzy-height`: the amount of fuzzy height boundary in the cropping trick. 
- `--output-path`: path to write overlayed detection or tracking result. If not specify, live video screen is shown
- `--output-fps`: the FPS to write the output video, if specified.  

