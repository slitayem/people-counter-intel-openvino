#!/bin/bash
set -o pipefail

# Script for starting all the components of the application

cecho(){
	RED="\033[0;31m"
	GREEN="\033[0;32m"
	YELLOW="\033[1;33m"
	NC="\033[0m" # No Color
	CYAN="\033[36m"
	printf "${!1}${2} ${NC}\n"
}

cecho "CYAN" "Starting all the application components: node server, web ui, ffmpeg server ..."

WORKSPACE="/home/openvino/people-counter"
if [ ! -d "$WORKSPACE/logs" ]; then
	mkdir $WORKSPACE/logs
fi

cd $WORKSPACE/webservice/server/node-server
nohup node ./server.js > $WORKSPACE/logs/mqtt-server.out 2> \
	$WORKSPACE/logs/mqtt-server.err < /dev/null &

cd $WORKSPACE/webservice/ui
yes | nohup npm run dev  > $WORKSPACE/logs/webservice-ui.out 2> \
	$WORKSPACE/logs/webservice-ui.err < /dev/null &

cd $WORKSPACE
nohup ffserver -f ./ffmpeg/server.conf  > $WORKSPACE/logs/ffserver.out 2> \
	$WORKSPACE/logs/ffserver.err < /dev/null &
