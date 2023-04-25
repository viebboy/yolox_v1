#!/bin/bash

model_conversion_tools="$(pwd)/src/model_conversion_tools"
onnx_dir="$(pwd)/src/onnx_models"
snpe_dir="$(pwd)/src/snpe_models"
container_name=model_conversion

if [ ! -d ${snpe_dir} ]; then
    mkdir ${snpe_dir} 
fi

docker run \
    -v ${model_conversion_tools}:/home/jenkins/model_conversion_tools \
    -v ${onnx_dir}:/home/jenkins/onnx_models \
    -v ${snpe_dir}:/home/jenkins/snpe_models \
    -v "$(pwd)/to_snpe.bash":/home/jenkins/to_snpe.bash \
    ${container_name} \
    bash /home/jenkins/to_snpe.bash
