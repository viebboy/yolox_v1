
#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <number_of_parallel_instances>"
  exit 1
fi

num_parallel_instances=$1

src="/home/jenkins/onnx_models"
dst="/home/jenkins/snpe_models"
total=$(find "$src" -type f -name '*.onnx' | wc -l)

cd /home/jenkins/model_conversion_tools

find "$src" -type f -name '*.onnx' -print0 |
  xargs -0 -P $num_parallel_instances -I{} sh -c '
    file="$1"
    filename=$(basename -- "$file")
    output_dlc="${2}/${filename%.onnx}.dlc"
    if [ ! -f "$output_dlc" ]; then
      python3 /home/jenkins/model_conversion_tools/convert.py --snpe-env snpe --snpe-fake-quantize ${file} "$output_dlc"
    fi
  ' _ {} "$dst"

