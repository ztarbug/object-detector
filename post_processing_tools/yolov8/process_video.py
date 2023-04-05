import argparse
import logging
from os.path import basename, isdir, isfile, join, splitext
import os

import cv2
import onnxruntime
import torch
import torchvision
from ultralytics import YOLO
import numpy as np

from wrapper import Wrapper

logging.basicConfig(format='%(asctime)s %(message)s')


def prepare_model(model_size: str):
    '''Check if the model file is in place, otherwise produce it'''

    model_file = f'yolov8{model_size}.pt'
    wrapped_model_file = f'yolov8{model_size}_wrapped.onnx'

    if isfile(wrapped_model_file):
        logging.info('Model is present.')
        return wrapped_model_file

    # Automatically downloads model if needed
    yolo = YOLO(model_file, task='detect')

    wrapped_model = Wrapper(yolo.model)

    # We need a valid model input for the exporter to run (says documentation...)
    input = torch.randint(0, 255, (1, 1080, 810,3), dtype=torch.uint8)

    logging.info(f'Writing model file {wrapped_model_file}')
    torch.onnx.export(
        wrapped_model,
        input,
        wrapped_model_file,
        input_names=['image'],
        output_names=['output'],
        dynamic_axes={
            'image': {1: 'height', 2: 'width'},
            'output': {0: 'num_boxes'},
        },
        opset_version=17,
    )

    return wrapped_model_file

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

    model = prepare_model(args.model_size)

    inf_session = onnxruntime.InferenceSession(model)

    cap = cv2.VideoCapture(args.videofile)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    out_file = prepare_out_file(args.videofile)
    
    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), src_fps, src_dimensions)

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if ret == True:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype('uint8')
            
            image_tensor_chw = torch.from_numpy(frame_rgb)
            image_tensor_bwhc = image_tensor_chw.permute((1, 0, 2)).unsqueeze(0)

            output = inf_session.run(
                None,
                {'image': image_tensor_bwhc.numpy()}
            )

            print(output)

