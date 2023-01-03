# assuming that this package has been installed with all dependencies

############################## config ########################## 
# data directory
DATA_DIR=./datasets
mkdir -p ${DATA_DIR}

# train annotation file
TRAIN_ANN=${DATA_DIR}/annotations/train.json
VAL_ANN=${DATA_DIR}/annotations/train.json

# data split
SPLIT=train

# min max area of bounding box
MIN_AREA=0
MAX_AREA=0.25

# exp file
CONFIG_FILE=./exps/open_image_person_detector_v2_exp7.py

# onnx file
ONNX_FILE=./YOLOX_outputs/open_image_person_detector_v2_exp7.onnx

# number of GPU
NB_GPU=4

# batch size
BATCH_SIZE=48

# batch size of the exported onnx model
INFER_BATCH_SIZE=1


############################# download & prepare dataset #######
python standalone/open_image_person_dataset.py --output-path ${DATA_DIR} --split ${SPLIT} --min-area ${MIN_AREA} --max-area ${MAX_AREA}


############################# train ############################
python -m yolox.tools.train \
    -f ${CONFIG_FILE} \
    -d ${NB_GPU} \
    -b ${BATCH_SIZE} \
    --fp16 \
    --data-dir ${DATA_DIR} \
    --train-ann ${TRAIN_ANN} \
    --val-ann ${VAL_ANN} \
    --cache

############################ export to ONNX ###################
python -m yolox.tools.export_onnx --output-name ${ONNX_FILE} --batch-size ${INFER_BATCH_SIZE} --exp-file ${CONFIG_FILE}
