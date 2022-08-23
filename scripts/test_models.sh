#!/bin/bash
set -o pipefail
## Script for downloading used models for the experiments

cecho(){
	RED="\033[0;31m"
	GREEN="\033[0;32m"
	YELLOW="\033[1;33m"
	NC="\033[0m" # No Color
	CYAN="\033[36m"
	printf "${!1}${2} ${NC}\n"
}

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <DEVICE> [Floating-point precision: FP16|FP32]"
  echo " e.g: $0 CPU FP16"
  exit 1;
fi

WORKSPACE="/home/openvino/people-counter"
./start_components.sh
sleep 10
cecho "CYAN" "All components were successfully started!"
OV_DIR="/opt/intel/openvino/"
MODEL_OPTIMIZER="/opt/intel/openvino/deployment_tools/model_optimizer"
MODELS_FOLDER="/home/openvino/app-artifacts/models"
THRESHOLD=0.8

DEVICE=$1
if [ -z ${2+x} ]; then
  DATA_TYPE="FP16"
else
  DATA_TYPE=$2
fi

echo "Tested model Floating-point precision $DATA_TYPE ..."

export MODELS_FOLDER=$MODELS_FOLDER
if [ ! -d "$MODELS_FOLDER" ]; then
	echo "Creating models folder ${MODELS_FOLDER}"
	mkdir -p $MODELS_FOLDER
else
	echo "Models folder ${MODELS_FOLDER} already exists"
fi

########################################
###### 1. Public Pre-Trained Models
########################################
# https://docs.openvino.ai/2021.4/omz_models_group_public.html
INTEL_MODELS=("ssd_mobilenet_v2_coco" \
	"ssdlite_mobilenet_v2")

for MODEL_NAME in ${INTEL_MODELS[@]}; do
	cecho "GREEN" "================================================"
	cecho "GREEN" "======  Public Model ${MODEL_NAME}"
	cecho "GREEN" "================================================"
	### 1. Downloading the model
	cecho "CYAN" "Downloading model files...."
	download_res=`python3 $OV_DIR/deployment_tools/tools/model_downloader/downloader.py \
		--name $MODEL_NAME -o $MODELS_FOLDER \
		| tee | grep 'Unpacking'`
	# Extract model full path from string like "Unpacking /home/openvino/models/public/ssdlite_mobilenet_v2/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz"
	MODEL_DIR=`echo $download_res | awk -F 'Unpacking' '{print $2}'| cut -d'.' -f 1`
	if [ "$MODEL_NAME" == "ssd_mobilenet_v2_coco" ]
	then
		MODEL_DIR=$MODEL_DIR"_2018_03_29"
	fi
	FILE="$MODEL_DIR/frozen_inference_graph.pb"
	[ $? -eq 0 ] && {
		echo "File $FILE exist."; eval mv "${FILE}"  $MODEL_DIR/$MODEL_NAME.pb;
	}
	### 2. Converting the model to IR format if needed
	####  Only needed for non-OpenVINO™ models
	if [ ! -d "$MODEL_DIR/$DATA_TYPE" ]; then
		cecho "CYAN" "Converting to IR format"
		python3 $MODEL_OPTIMIZER/mo.py \
			--input_model $MODEL_DIR/$MODEL_NAME.pb \
			--tensorflow_object_detection_api_pipeline_config $MODEL_DIR/pipeline.config \
			--reverse_input_channels \
			--transformations_config $MODEL_OPTIMIZER/extensions/front/tf/ssd_v2_support.json \
			--data_type=$DATA_TYPE --output_dir $MODEL_DIR/$DATA_TYPE
	else
		cecho "IR format of the model already exists: $MODEL_DIR/$DATA_TYPE"
	fi

	### 3. Running the people counter service
	python3 ../src/main.py -i ../resources/Pedestrian_Detect_2_1_1.mp4 --debug \
		-m ${MODEL_DIR}/${DATA_TYPE}/${MODEL_NAME}.xml -d $DEVICE \
		-pt $THRESHOLD -n ${MODEL_NAME} -dt ${DATA_TYPE} | \
		ffmpeg -v warning -f rawvideo -pixel_format rgb24 \
		-video_size 768x432 -framerate 24 -i - -threads 1 -c copy  \
		http://0.0.0.0:3004/fac.ffm
done

##################################################
###### 2. Intel pre-trained Models
##################################################
# https://docs.openvino.ai/2021.4/omz_models_group_intel.html
PUBLIC_MODELS=("pedestrian-detection-adas-0002" \
	"person-detection-retail-0013")

for i in  ${!PUBLIC_MODELS[@]}; do
	MODEL_NAME=${PUBLIC_MODELS[$i]}
	echo "================================================"
	echo "GREEN" "======  Intel Model ${MODEL_NAME}"
	echo "================================================"
	### 1. Downloading the model
	echo "Downloading model files...."
	python3 $OV_DIR/deployment_tools/tools/model_downloader/downloader.py  --name $MODEL_NAME -o $MODELS_FOLDER
	MODEL_DIR=$MODELS_FOLDER/intel/$MODEL_NAME
	
	### 2. Running the people counter service
	python3 ../src/main.py -i ../resources/Pedestrian_Detect_2_1_1.mp4 --debug \
		-m $MODEL_DIR/$DATA_TYPE/${MODEL_NAME}.xml -d $DEVICE \
		-pt $THRESHOLD -n $MODEL_NAME -dt $DATA_TYPE | \
		ffmpeg -v warning -f rawvideo -pixel_format rgb24 \
		-video_size 768x432 -framerate 24 -i - -threads 0 -c copy  \
		http://0.0.0.0:3004/fac.ffm
done	

##################################################
###### 3. Converting Tensorflow model(s)
##################################################
# https://docs.openvino.ai/2021.4/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html
# https://docs.openvino.ai/cn/2021.4/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html
# MODEL_NAME="faster_rcnn_inception_resnet_v2"
# MODEL_DIR=$MODELS_FOLDER/tensorflow/$MODEL_NAME
# wget http://download.tensorflow.org/models/object_detection/tf2/20200711/$MODEL_NAME_640x640_coco17_tpu-8.tar.gz
# tar xvf ${MODEL_DIR}.tar.gz
# mv ${MODEL_DIR}/frozen_inference_graph.pb  $MODEL_DIR/$MODEL_NAME.pb
# ### Converting the TF2 model to IR format
# python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
# 	--saved_model_dir ${MODEL_DIR}/saved_model \
# 	--tensorflow_object_detection_api_pipeline_config ${MODEL_DIR}/pipeline.config \
# 	--reverse_input_channels \
# 	--data_type=$DATA_TYPE --input_shape "[1,640,640,3]" \
# 	--output_dir ${MODEL_DIR}/${DATA_TYPE} \
# 	--transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support_api_v2.0.json

# python3 ../src/main.py -i ../resources/Pedestrian_Detect_2_1_1.mp4 --debug \
# 	-m ${MODEL_DIR}/${DATA_TYPE}/saved_model.xml -d $DEVICE \
# 	-pt $THRESHOLD -n ${MODEL_NAME} -dt ${DATA_TYPE} | \
# 	ffmpeg -v warning -f rawvideo -pixel_format rgb24 \
# 	-video_size 768x432 -framerate 24 -i - -threads 1 -c copy  \
# 	http://0.0.0.0:3004/fac.ffm
