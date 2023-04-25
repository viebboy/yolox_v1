#!/bin/bash

src="/home/jenkins/onnx_models"
dst="/home/jenkins/snpe_models"
count=0
total=$(find "$src" -type f -name '*.onnx' | wc -l)

cd /home/jenkins/model_conversion_tools

time {
  for file in "$src"/*.onnx; do
    filename=$(basename -- "$file")
    python3 /home/jenkins/model_conversion_tools/convert.py --snpe-env snpe --snpe-fake-quantize ${file} "${dst}/${filename%.onnx}.dlc"
    ((count++))
    percent=$((count * 100 / total))
    echo "Progress: $percent% ($count/$total files processed)"
  done
}
