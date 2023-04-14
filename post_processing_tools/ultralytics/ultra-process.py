#!/usr/bin/env python

import sys
import os
import datetime
import argparse
import logging
import signal

from pprint import pprint
from ultralytics import YOLO
import cv2

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()

running = True

task_suffixes = {
    'detect': '',
    'segment': '-seg',
    'classify': '-cls',
    'pose': '-pose',
}

def do_inferencing(source, model_size, task, preview):
    global running
    if source.isdigit():
        source = int(source)
    cap    = cv2.VideoCapture(source)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    start_time = datetime.datetime.now()
    time_prefix = start_time.strftime('%Y-%m-%dT%H-%M-%S')
    output_filename = os.path.join('out', f'{task}-{model_size}-{time_prefix}-{source_to_filename(source)}.mp4')
    logger.info(f'Writing to output file: {output_filename}')
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    if preview:
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", 800, 600)

    model = YOLO(model_name(model_size, task))
    while cap.isOpened() and running:
        ret, frame = cap.read()
        if ret == True:
            prediction = model.predict(frame)
            res_plotted = prediction[0].plot()
            
            if preview:
                cv2.imshow('output', res_plotted)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break 

            out.write(res_plotted)
        else:
            logger.error("couldn't read next frame")
            break
    
    end_time = datetime.datetime.now()
    logger.info(f'Finished post processing in {end_time - start_time}s')
    cap.release()
    cv2.destroyAllWindows()

def model_name(size, task):
    suffix = task_suffixes[task]
    return f'yolov8{size}{suffix}.pt'

def source_to_filename(source):
    # based on which kind of source an according output filename is created here
    if isinstance(source, int):
        return "USB-CAM-" + str(source) + ".mp4"    
    if 'rtsp' in source:
        source = source.split('@')[1]
        source = source.split(':')[0]
        return "rtsp-" + source + ".mp4"
    if os.path.exists(source) and not os.path.isdir(source):
        return os.path.basename(source)
    return "unknown-source.mp4"

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('videosource', help='any video source opencv understands, e.g. 0,1,... for usb cams, "rtsp://..." for RTSP streams, /path/video.mp4 for video file')
    arg_parser.add_argument('-m', '--model-size', choices=['n', 's', 'm', 'l', 'x'], default='n', help='the size of the model to use (nano, small, medium, large, xlarge); defaults to "nano"')
    arg_parser.add_argument('-t', '--task', choices=['detect', 'segment', 'classify', 'pose'], default='detect', help='the task to perform; defaults to "detect"')
    arg_parser.add_argument('-p', '--preview', action='store_true', help='whether to show a live preview window, reduces performance slightly. If the window is in focus, press "q" to exit.')
    args = arg_parser.parse_args()

    def signal_handler(signum, _):
        signame = signal.Signals(signum).name
        logger.warning(f'Received {signame}. Exiting...')
        global running
        running = False

    logger.info("Setting up signal handlers...")
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    if not os.path.isdir('out'):
        os.mkdir('out')

    do_inferencing(args.videosource, args.model_size, args.task, args.preview)