#!/bin/bash 

src="/home/jenkins/onnx_models"
dst="/home/jenkins/snpe_models"

for file in "$src"/*.onnx; do
    filename=$(basename -- "$file")
    python3 /home/jenkins/model_conversion_tools/convert.py --snpe-env snpe --snpe-fake-quantize ${file} "${dst}/${filename%.onnx}.dlc"
done
