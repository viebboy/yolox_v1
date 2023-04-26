
#!/bin/bash


src="/home/jenkins/onnx_models"
dst="/home/jenkins/snpe_models"
total=$(find "$src" -type f -name '*.onnx' | wc -l)

cd /home/jenkins/model_conversion_tools

find "$src" -type f -name '*.onnx' -print0 |
  xargs -0 -P 4 -I{} sh -c '
    file="$1"
    filename=$(basename -- "$file")
    output_dlc="${2}/${filename%.onnx}.dlc"
    if [ ! -f "$output_dlc" ]; then
      python3 /home/jenkins/model_conversion_tools/convert.py --snpe-env snpe --snpe-fake-quantize ${file} "$output_dlc"
    fi
  ' _ {} "$dst"

