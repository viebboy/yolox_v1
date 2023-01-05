# assuming that this package has been installed with all dependencies

############################## config ########################## 
# model name
MODEL_NAME=yolox_x
# data directory
DATA_DIR=/mnt/datasets
mkdir -p ${DATA_DIR}

# train annotation file
TRAIN_ANN=train.json
VAL_ANN=train.json

# data split
SPLIT=train

# min max area of bounding box
MIN_AREA=0
MAX_AREA=0.25

# exp file
CONFIG_FILE=./exps/open_image_person_detector_${MODEL_NAME}.py

# onnx file
ONNX_FILE=./YOLOX_outputs/open_image_person_detector_${MODEL_NAME}.onnx

# number of GPU
NB_GPU=1

# batch size
BATCH_SIZE=12

# batch size of the exported onnx model
INFER_BATCH_SIZE=1

# pretrained checkpoint, please replace with the correct checkpoint that corresponds to the CONFIG_FILE
CHECKPOINT_LINK=https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/${MODEL_NAME}.pth
CHECKPOINT_PATH=YOLOX_outputs/${MODEL_NAME}.pth


############################# download & prepare dataset #######
python standalone/open_image_person_dataset.py --output-path ${DATA_DIR} --split ${SPLIT} --min-area ${MIN_AREA} --max-area ${MAX_AREA}


############################# download checkpoint ##############
mkdir -p YOLOX_outputs
wget ${CHECKPOINT_LINK} -O ${CHECKPOINT_PATH}


############################# train ############################
python -m yolox.tools.train \
    -f ${CONFIG_FILE} \
    -d ${NB_GPU} \
    -b ${BATCH_SIZE} \
    --fp16 \
    --data-dir ${DATA_DIR} \
    --train-ann ${TRAIN_ANN} \
    --val-ann ${VAL_ANN} \
    -c ${CHECKPOINT_PATH} \
    --cache


############################ export to ONNX ###################
python -m yolox.tools.export_onnx --output-name ${ONNX_FILE} --batch-size ${INFER_BATCH_SIZE} --exp-file ${CONFIG_FILE}
