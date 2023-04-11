import argparse
import logging
from os.path import basename, isdir, isfile, join, splitext
import os

import cv2
import onnxruntime
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes
from ultralytics import YOLO
import numpy as np
import json

from wrapper import Wrapper

logging.basicConfig(format='%(asctime)s - %(message)s')

def get_classes():
    try:
        with open('yolov8_classes.json', 'r') as f:
            js = json.load(f)
            return {int(k):v for k,v in js.items()}
    except OSError as e:
        logging.error(f'Could not load yolo classes from file. Exception: {e}')
        exit(1)

def prepare_model(model_size: str):
    '''Check if the model file is in place, otherwise produce it'''

    model_file = f'yolov8{model_size}.pt'

    # Automatically downloads model if needed
    yolo = YOLO(model_file, task='detect')

    wrapped_model = Wrapper(yolo.model)

    return wrapped_model

def prepare_out_file(src_file):
    src_file_base = splitext(basename(src_file))[0]
    out_file = f'{join("out", src_file_base)}_annotated.avi'

    if not isdir('out'):
        os.mkdir('out')
    
    return out_file
    

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('videofile')
    arg_parser.add_argument('-m', '--model-size', choices=['n', 's', 'm', 'l', 'x'], default='n')
    args = arg_parser.parse_args()

    classes = get_classes()
    model = prepare_model(args.model_size)
    print(model)

    cap = cv2.VideoCapture(args.videofile)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    out_file = prepare_out_file(args.videofile)
    
    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), src_fps, src_dimensions)

    current_frame = 0
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if ret == True:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype('uint8')
            
            image_tensor_hwc = torch.from_numpy(frame_rgb)
            image_tensor_bwhc = image_tensor_hwc.permute((1, 0, 2)).unsqueeze(0)

            output = model.forward(image_tensor_bwhc)

            boxes = output[:, :4]
            labels = [
                f'{classes[int(class_id)]} - {round(float(confidence)*100,2)}%'
                for confidence, class_id in output[:, 4:]
            ]

            image_tensor_chw = image_tensor_hwc.permute((2,0,1))
            print(boxes)
            annotated_image = draw_bounding_boxes(image_tensor_chw, boxes, width=4, labels=labels)

            torchvision.io.write_jpeg(annotated_image, f'out/{current_frame}.jpg')

            current_frame += 1

