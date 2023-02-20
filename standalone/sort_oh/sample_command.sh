# assuming all artifacts are in ./data, including the sample video
# this is the command to run on a video using tracker v2 (with visual embedding)
#
python tracker_v2.py --tracker-config ./config_v2.json \
                     --input-type video \
                     --wide-angle False \
                     --detector-model-file ./data/v2_exp7_bs=1.onnx \
                     --embedding-model-file ./data/osnet_ain_x1_0.onnx \
                     --input-path ./data/MOT17-09-SDP-raw.mp4 \
                     --output-path ./data/MOT17-09-SDP-raw_with_visual_embedding.mp4 \
                     --output-fps 5

#python tracker_v1.py --tracker-config ./config_v1.json \
#                     --input-type video \
#                     --wide-angle False \
#                     --onnx-file ./data/v2_exp7_bs=1.onnx \
#                     --input-path ./data/MOT17-09-SDP-raw.mp4 \
#                     --output-path ./data/MOT17-09-SDP-raw_without_visual_embedding.mp4 \
#                     --output-fps 5
