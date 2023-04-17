#!/usr/bin/env python

import argparse
import sys
import os
import signal
import datetime

import logging

from ultralytics import YOLO
import cv2
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()

yolov8_model = 'yolov8n-pose.pt'
running = True

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('videosource', help='any video source opencv understands, e.g. 0,1,... for usb cams, "rtsp://..." for RTSP streams, /path/video.mp4 for video file')
    arg_parser.add_argument('-m', '--model-size', choices=['n', 's', 'm', 'l', 'x'], default='n', help='the size of the model to use (nano, small, medium, large, xlarge); defaults to "nano"')
    arg_parser.add_argument('-p', '--preview', action='store_true', help='whether to show a live preview window, reduces performance slightly. If the window is in focus, press "q" to exit.')
    arg_parser.add_argument('-l', '--log-classes', type=lambda s: s.split(','), dest='class_list', help='for every frame, log detected objects of these comma-delimited classes (class name must match a class in "yolov8_classes.json") into a sidecar .csv file')
    arg_parser.add_argument('-o', '--output-path', required=False, default="out", help='where to output processed video files')
    global args
    args = arg_parser.parse_args()

    capture_abort()

    check_output_path(args.output_path)
    do_inferencing(args)

def capture_abort():
    def signal_handler(signum, _):
        signame = signal.Signals(signum).name
        logger.warning(f'Received {signame}. Exiting...')
        global running
        running = False

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)    

def check_output_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        if not os.path.isdir(path):
            print("output folder is file, can't output results. Exit")
            sys.exit() 

def do_inferencing(args):
    source = args.videosource
    if source.isdigit():
        source = int(source)

    start_time = datetime.datetime.now()
    time_prefix = start_time.strftime("%Y%m%d-%H%M")

    output_filename = create_ouput_filename(source)
    output_filename = 'out/' + time_prefix + "-" + yolov8_model + "-" + output_filename

    cap    = cv2.VideoCapture(source)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    if args.preview:
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", 1024, 768)

    model = YOLO(yolov8_model)
    while(cap.isOpened() and running):
        ret, frame = cap.read()
        if ret == True:
            #res_plotted = pose_objects(frame, model)
            prediction = model.predict(frame)
            res_plotted = prediction[0].plot()
            
            if args.preview:
                cv2.imshow("output", res_plotted)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break 
            out.write(res_plotted)
        else:
            break
    
    end_time = datetime.datetime.now()
    print("Finished post processing in " + str(end_time - start_time))

    cap.release()
    cv2.destroyAllWindows()

def create_ouput_filename(source):
    # based on which kind of source an according output filename is created here
    if  isinstance(source, int):
        return "USB-CAM-" + str(source) + ".mp4"    
    if 'rtsp' in source:
        source = source.split('@')[1]
        source = source.split(':')[0]
        return "rtsp-" + source + ".mp4"
    if os.path.exists(source) and not os.path.isdir(source):
        return os.path.basename(source)
    return "unknown-source.mp4"

def pose_objects(frame, model):

    overlay = frame.copy() # overlay to add segmentation polygons
    alpha = 0.4  # Transparency factor.
    pose = model.predict(frame)
    names = pose[0].names
    logger.info(pose[0])

    # todo
    img_out = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return img_out

main()