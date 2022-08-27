"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Code updated by @Saloua Litayem
"""

import os
import sys
import time
import socket
import json
import logging as log
from argparse import ArgumentParser

from webcolors import name_to_rgb
import cv2
import numpy as np

from inference import Network
import utils
import images
import logger

# MQTT server environment variables
HOSTNAME = socket.gethostname()
MQTT_HOST = socket.gethostbyname(HOSTNAME)
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

APP_DIR = os.path.expandvars("$HOME/app-artifacts")
LOGS_DIR = os.path.join(APP_DIR, "logs")
IMAGES_EXTENSIONS = ['.jpg', '.bmp']
MODEL_NAME = os.getenv('MODEL_NAME')
PERF_FOLDER = os.path.join(APP_DIR, "perf")


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-n", "--name", required=True, type=str,
                        help="Model name: needed for performance export.")
    parser.add_argument("-dt", "--data_type", required=True, type=str,
                    help="Floating-point precision e.g. FP32")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.7,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-db", "--debug", action='store_true', default=False,
                    help="Set to use the app in debug mode."
                    "(False by default)")
    return parser


def process_input(image, shape, infer_network, prob_threshold=0.7):
    """
    Detect people in provided image and draw a bounding box around the detected person object.
    The resulting image is written back to stdout
    """
    ### Pre-process the image as needed ###
    original_height, original_width, _ = image.shape
    image_ = images.preprocess(image, shape)

    ### Start asynchronous inference for specified request ###
    inference_start = time.time()
    infer_request_handle = infer_network.exec_net(image_, 0)

    ### Wait for the result ###
    outputs = infer_network.wait()

    if outputs:
        inference_duration = time.time() - inference_start
        ### Get the results of the inference request ###
        inference_outputs = infer_network.get_output(request_id=0)

        image_, current_count, boxes, confs = images.process_ssd_output(
            image, inference_outputs, prob_threshold, original_width, original_height)
    return image_, current_count, inference_duration


def infer_on_stream(args, mqtt_client):
    """
    Initialize the inference network, capture video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param mqtt_client: MQTT mqtt_client
    :return: None
    """
    log.info("Infering on stream ....")
    try:
        prob_threshold = args.prob_threshold
        infer_network = Network()
        MODEL_NAME = os.getenv('MODEL_NAME', args.name)
        ### Load the model
        model, input_shape = infer_network.load_model(args.model, args.device, 1)
        model_size = utils.human_readable_size(sys.getsizeof(model))
        #### Perform inference on the input Video Capture
        # batch_size, channels, inp_height, inp_width = input_shape
        log.info(f"Model Input shape {input_shape}")

        #### SINGLE IMAGE Mode
        single_image_mode = list(filter(args.input.endswith, IMAGES_EXTENSIONS)) != []
        log.debug(f"Single Image Mode: {single_image_mode}")
        if single_image_mode:
            image = process_input(cv2.imread(args.input), input_shape, infer_network, prob_threshold)
            cv2.imwrite(
                os.path.join("perf", f"{args.data_type}_people_counter_{MODEL_NAME}.jpg"), image)
            return

        #### VIDEO capture mode
        start_time = time.time()
        ### Handle the input video_capture ###
        video_capture = cv2.VideoCapture(args.input)
        video_capture.open(args.input)
        if not video_capture.isOpened():
            log.exception(f"Error opening input file (video or image {args.input})")
            exit(1)

        capture_info = {
            "framecount": video_capture.get(cv2.CAP_PROP_FRAME_COUNT),
            "fps": video_capture.get(cv2.CAP_PROP_FPS),
            "width": int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "codec": int(video_capture.get(cv2.CAP_PROP_FOURCC))
        }

        log.info(f"Video Captute info {json.dumps(capture_info, indent=4)}")

        total_count = 0
        current_count = 0
        last_count = 0
        person_start_time = 0
        inference_durations = []

        # Create VideoWriter object with X264 codec
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(
            os.path.join(
                PERF_FOLDER,
                f"{args.data_type}_{args.device}_{MODEL_NAME}.avi"
            ), fourcc, 20.0, (capture_info['width'], capture_info['height'] ))

        ### Loop until video_capture is over
        while video_capture.isOpened():
             ### Reading from the video capture
            (flag, input_image) = video_capture.read()
            if not flag:
                log.debug("End of video_captureed content was reached.")
                break
            key_pressed = cv2.waitKey(60)

            image, current_count, inference_duration = process_input(
                input_image, input_shape, infer_network,
                prob_threshold)

            cv2.putText(
                image, f"Inference Time: {(inference_duration * 1000):.2f}ms", (15, 15),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, name_to_rgb('green'), 1, cv2.LINE_AA)

            inference_durations.append(inference_duration)

            # log.debug(f"Current Count {current_count}")

            # Assumption: In each frame, we have only one person entering the scene
            # A person was detected in the frame compared to previously processed frame
            if current_count > last_count:
                total_count += current_count - last_count
                person_start_time = time.time()
                log.debug("New Person coming")

            # There is no person detected in the newframe
            if current_count < last_count:
                duration = int((time.time() - person_start_time) * 1000)
                # Handle false negative detections by not including very short durations
                # It is counter intuitive that a person stays in the scene for a duration
                # shorter than 2 seconds
                if duration > 1000:
                    log.debug(f"A Person left {duration}")
                    utils.publish_messages(mqtt_client, {
                        "person/duration": {"duration": duration},
                    })
            
            utils.publish_messages(mqtt_client, {"person": {"count": current_count}})

            last_count = current_count
            # Output the updated image to stdout
            # which will be piped out to the FFMPEG servers
            images.bufferize(image, capture_info['width'], capture_info['height'])

            # output video update
            out.write(image)

        total_duration = time.time() - start_time
        avg_time = np.mean(inference_durations)
        perf_metadata = {}
        with open(os.path.join(PERF_FOLDER,
            f"{args.data_type}_{args.device}_{prob_threshold}_{MODEL_NAME}.json"), "w") as file_:
            perf_metadata = {
                "Model Name": MODEL_NAME,
                "People count": total_count,
                "AVG inference time (sec.)": round(avg_time, 3),
                "Completion Time (Sec.)": round(total_duration, 3),
                "Detection confidence threshold (%)": prob_threshold * 100,
                "Floating-point precision": f"{args.data_type}",
                "Loaded Model Size": model_size,
            }
            json.dump(perf_metadata, file_, indent=4)
        log.info(json.dumps(perf_metadata, indent=4))

        if mqtt_client:
            mqtt_client.disconnect()

        video_capture.release()
        out.release()
        cv2.destroyAllWindows()

    except Exception as exp:
        logger.log_exception(exp)


def main():
    """
    Load the network and parse the output.
    """
    args = build_argparser().parse_args()

    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR, exist_ok=True)
    if not os.path.exists(PERF_FOLDER):
        os.makedirs(PERF_FOLDER, exist_ok=True)

    logger.SetupLogging().setup_logging(
        default_level=log.DEBUG if args.debug else log.INFO)
    log.info(" Starting People Counter....")

    # Connect to the MQTT server
    mqtt_client = utils.connect_mqtt(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    log.info("Successfully connected to MQTT")
    infer_on_stream(args, mqtt_client)


if __name__ == '__main__':
    main()
