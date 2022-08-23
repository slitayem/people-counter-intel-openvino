#!/bin/bash
set -o pipefail

cecho(){
	# colored echo function
	RED="\033[0;31m"
	GREEN="\033[0;32m"
	YELLOW="\033[1;33m"
	NC="\033[0m" # No Color
	CYAN="\033[36m"
	printf "${!1}${2} ${NC}\n"
}

WORKSPACE="/home/openvino"
./start_components.sh
sleep 10

export OV_DIR=/opt/intel/openvino/
export MODEL_OPTIMIZER=/opt/intel/openvino/deployment_tools/model_optimizer

#### 1. Download pedestrian-detection-adas-0002 model
export MODEL_NAME="pedestrian-detection-adas-0002"
cecho "YELLOW" "=========================================="
cecho "YELLOW" "====== Model $MODEL_NAME"
cecho "YELLOW" "=========================================="

DATA_TYPE=FP32
DEVICE=CPU
THRESHOLD=0.8
WORKSPACE="/home/openvino/people-counter"
export MODELS_FOLDER="/home/openvino/app-artifacts/models"
if [ ! -d "$MODELS_FOLDER" ]; then
	mkdir -p $MODELS_FOLDER
fi

cecho "GREEN" "Downloading the model files ..."
python3 $OV_DIR/deployment_tools/tools/model_downloader/downloader.py  --name $MODEL_NAME -o $MODELS_FOLDER
MODEL_DIR=$MODELS_FOLDER/intel/$MODEL_NAME

cecho "GREEN" "Running the people counter service ..."
python3 ../src/main.py -i ../resources/Pedestrian_Detect_2_1_1.mp4 --debug \
	-m $MODEL_DIR/$DATA_TYPE/$MODEL_NAME.xml -d $DEVICE \
	-pt $THRESHOLD -n $MODEL_NAME -dt $DATA_TYPE | \
	ffmpeg -v warning -f rawvideo -pixel_format rgb24 \
	-video_size 768x432 -framerate 24 -i - -threads 2 -c copy  \
	http://0.0.0.0:3004/fac.ffm
