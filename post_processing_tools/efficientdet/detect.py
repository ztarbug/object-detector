import os
import sys
import tarfile
import time
import datetime

from urllib import request

import pprint as pp

import cv2
import tensorflow as tf

PERSON_CLASS = 1
BICYCLE_CLASS = 2
CAR_CLASS = 3
BUS_CLASS = 6
TRUCK_CLASS = 8

my_objects = {1:"Person", 2:"Bicycle", 3:"Car", 6:"Bus", 8:"Truck"}

model_path = 'model'
output_path = 'out'
efficient_det_version = 4

def main():
    check_args()
    target_file = sys.argv[1]
    check_project_structure(target_file)
    model = download_tf_model(efficient_det_version)
    do_inferencing(model, target_file)

def check_args():
    if len(sys.argv) < 2:
        print("provide file to be processed")
        sys.exit()

def check_project_structure(target_file):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    else:
        if not os.path.isdir(model_path):
            print("model folder is file, can't output results. Exit")
            sys.exit()
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        if not os.path.isdir(output_path):
            print("output folder is file, can't output results. Exit")
            sys.exit() 
    
    if not os.path.exists(target_file):
        print("provided target file " + target_file + " does not exist. Exit")
        sys.exit()

def download_tf_model(efficientDet_version):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
    file_name = 'efficientdet_d#_coco17_tpu-32.tar.gz'.replace('#',str(efficientDet_version))
    download_url = base_url + file_name

    target_file = model_path + '/efficientdet_d' + str(efficientDet_version) + ".tar.gz"
    if not os.path.exists(target_file):
        request.urlretrieve(download_url, target_file)
    else:
        print("model already downloaded")
    
    tar = tarfile.open(target_file)
    tar.extractall(model_path)
    tar.close()

    extracted_model_path = model_path + '/' + file_name.split('.')[0] + '/saved_model'
    return extracted_model_path

def detect_objects(frame, efficientdet):
    frame = tf.constant(frame, dtype=tf.uint8)
    frame = tf.expand_dims(frame, 0)
    
    prediction = efficientdet(frame)
    boxes = prediction['detection_boxes']
    scores = prediction['detection_scores']
    classes = prediction['detection_classes']

    if len(classes.shape) > 1:
        scores = tf.squeeze(scores)
        boxes = tf.squeeze(boxes)
        classes = tf.squeeze(classes)
    
    return boxes, scores, classes  

def do_inferencing(downloaded_model_path, video_path):

    efficientdet = tf.saved_model.load(downloaded_model_path)

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    start_time = datetime.datetime.now()

    output_filename = os.path.basename(video_path)
    time_prefix = start_time.strftime("%Y%m%d-%H%M")
    out = cv2.VideoWriter('out/' + time_prefix + "-ED-" + str(efficient_det_version) + "-" + output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    #cv2.namedWindow("output", cv2.WINDOW_NORMAL)

    f = 1
    all_time_statistics = {}
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            start = time.time()
            boxes, scores, classes = detect_objects(frame, efficientdet)

            det_results = "Detected: "
            object_count = {}
            for i, box in enumerate(boxes):
                score = float(scores[i])
                if score >= 0.3:
                    class_id = int(classes[i])
                    if class_id in my_objects:
                        name=my_objects[class_id]
                        if name in object_count:
                            object_count[name] += 1
                        else:
                            object_count[name] = 1
                    else:
                        break
                    y2, x1, y1, x2 = box
                    x1 = int(tf.keras.backend.get_value(x1) * width)
                    x2 = int(tf.keras.backend.get_value(x2) * width)
                    y1 = int(tf.keras.backend.get_value(y1) * height)
                    y2 = int(tf.keras.backend.get_value(y2) * height)
                    det_results += " " + name + "-" + str(round(score, 2)) + ":(" + str(x1) + "," + str(y1) + ")-(" + str(x2) + "," + str(y2) + "),  "

                    font_scale = max(x2-x1, y2-y1) / 200.0
                    thickness = int(max(x2-x1, y2-y1) / 100.0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), thickness)
                    cv2.putText(
                        img = frame,
                        text = name + ": " + str(round(score, 2)),
                        org = (x1+3,y1-5),
                        fontFace = cv2.FONT_HERSHEY_DUPLEX,
                        fontScale = font_scale,
                        color = (125, 246, 55),
                        thickness = 3
                    )
                all_time_statistics = create_overall_statistics(all_time_statistics, object_count)
                draw_statistics(frame, all_time_statistics, object_count)
            end = time.time()

            out.write(frame)
            duration = end-start
            print("Frame " + str(f) + "/" + str(length) + " (" +  str(round(duration,3)) + "s) " + str(length - f)  + " | " + det_results + "\n")
            f += 1
            #cv2.imshow("output",frame)
            # Press Q on keyboard to  exit
            #if cv2.waitKey(25) & 0xFF == ord('q'):
            #    break        
        else: 
            break
    end_time = datetime.datetime.now()
    print(end_time - start_time)

    cap.release()
    cv2.destroyAllWindows()

def create_overall_statistics(all, current):
    for object in current:
        if not object in all:
            all[object] = current[object]
        else:
            if all[object] < current[object]:
                all[object] = current[object]
    
    return all

def draw_statistics(frame, all, current):
    first_column = 10
    second_column = 290
    cv2.putText(
        img = frame,
        text = "Statistics Frame",
        org = (first_column,50),
        fontFace = cv2.FONT_HERSHEY_DUPLEX,
        fontScale = 1.0,
        color = (125, 246, 55),
        thickness = 2
    )
    offset = 30
    for object in current:
        cv2.putText(
            img = frame,
            text = object + ": " + str(current[object]),
            org = (first_column,50 + offset),
            fontFace = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 1.0,
            color = (125, 246, 55),
            thickness = 2
        )
        offset += 30

    cv2.putText(
        img = frame,
        text = "Statistics Overall",
        org = (second_column,50),
        fontFace = cv2.FONT_HERSHEY_DUPLEX,
        fontScale = 1.0,
        color = (30, 30, 30),
        thickness = 2
    )
    offset = 30
    for object in all:
        cv2.putText(
            img = frame,
            text = object + ": " + str(all[object]),
            org = (second_column,50 + offset),
            fontFace = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 1.0,
            color = (30, 30, 30),
            thickness = 2
        )
        offset += 30

main()