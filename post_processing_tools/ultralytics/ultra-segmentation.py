#!/usr/bin/env python

import argparse
import os
import sys
import signal
import datetime

import logging

from ultralytics import YOLO
import cv2
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()

running = True
class_list = [0,1,2]
heatmap_colors = {
    0: ["PERSON", (102, 204, 0)], #(0, 204, 102)
    1: ["BICYCLE", (255, 153, 102)],
    2: ["CAR", (255, 0, 0)],
    5: ["BUS", (0, 0, 255)],   
    7: ["TRUCK", (153, 102, 255)]
}

dense_map = None

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('videosource', help='any video source opencv understands, e.g. 0,1,... for usb cams, "rtsp://..." for RTSP streams, /path/video.mp4 for video file')
    arg_parser.add_argument('-m', '--model-size', choices=['n', 's', 'm', 'l', 'x'], default='n', help='the size of the model to use (nano, small, medium, large, xlarge); defaults to "nano"')
    arg_parser.add_argument('-p', '--preview', action='store_true', help='whether to show a live preview window, reduces performance slightly. If the window is in focus, press "q" to exit.')
    arg_parser.add_argument('-o', '--output-path', required=False, default="out", help='where to output processed video files')
    arg_parser.add_argument('-heat', '--heatmap', required=False, default=False, help='create a heat map of detected objects')
    global args
    args = arg_parser.parse_args()

    capture_abort()

    global yolov8_model
    yolov8_model = f'yolov8{args.model_size}-seg.pt'

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
    output_filename = args.output_path + "/" + time_prefix + "-" + yolov8_model + "-" + output_filename

    cap    = cv2.VideoCapture(source)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))    
    if args.preview:
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", 1024, 768)
    
    model = YOLO(yolov8_model)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            prediction = model.predict(frame)
            res_plotted = segment_objects(frame, prediction) # if you want to work with prediction result
            #res_plotted = prediction[0].plot()            
            if args.heatmap:
                update_heatmap(res_plotted, prediction[0])
                #res_plotted = res_plotted + 0.33 * dense_map
                alpha = 0.1
                res_plotted = cv2.addWeighted(dense_map.astype(np.uint8), alpha, res_plotted, 1-alpha,0)
                #res_plotted[res_plotted > 255] = 255 # prevent color values > 255

            if args.preview:
                cv2.imshow('output', res_plotted)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break 
            #out.write(res_plotted) 
        else:
            break

    end_time = datetime.datetime.now()
    print("Finished post processing in " + str(end_time - start_time))
    cap.release()
    cv2.destroyAllWindows()

def update_heatmap(frame, prediction):
    global dense_map

    if(dense_map is None):
        dense_map = np.zeros_like(frame)
    
    if prediction.masks is not None:
        for idx, mask in enumerate(prediction.masks.xy):
            conf = prediction.boxes[idx].conf.data[0].item()
            if conf > 0.35:    
                polyline = mask.astype(np.int32)
                predicted_class_id = prediction.boxes[idx].cls.data[0].item()
                if predicted_class_id in class_list:
                    x,y = centroid(polyline)
                    color = heatmap_colors[predicted_class_id][1]
                    heatmap_entry = np.zeros_like(dense_map)
                    heatmap_entry = cv2.circle(heatmap_entry, (int(x),int(y)), 10, color=color, thickness=-1)
                    dense_map = dense_map + 0.1 * heatmap_entry # add layer for latest objects
                    dense_map[dense_map > 255] = 255 # addition will eventuall reach max color level


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

def segment_objects(frame, seg):

    overlay = frame.copy() # overlay to add segmentation polygons
    alpha = 0.4  # Transparency factor.
    names = seg[0].names    

    for idx, mask in enumerate(seg[0].masks.xy):
        conf = seg[0].boxes[idx].conf.data[0].item()
        if conf > 0.3:
            polyline = mask.astype(np.int32)
            predicted_class_id = seg[0].boxes[idx].cls.data[0].item()
            if predicted_class_id in class_list:
                pred_class = names[predicted_class_id]
                x,y = centroid(polyline)
                COLOR = (100,int(255/(predicted_class_id+1)),int(255/(predicted_class_id+1)))
                COLOR2 = (5,int(255/(predicted_class_id+1)),int(255/(predicted_class_id+1)))
                cv2.fillPoly(overlay,[polyline], COLOR)
                cv2.putText(
                    overlay, 
                    f'{pred_class} ({round(float(conf)*100,2)}%)', 
                    (round(x), round(y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, 
                    color=COLOR2, 
                    thickness=5)

    img_out = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return img_out

def centroid(vertexes):
     _x_list = [vertex [0] for vertex in vertexes]
     _y_list = [vertex [1] for vertex in vertexes]
     _len = len(vertexes)
     _x = sum(_x_list) / _len
     _y = sum(_y_list) / _len
     return(_x, _y) 

if __name__ == '__main__':
    main()