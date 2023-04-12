import os
import sys
import datetime

from pprint import pprint
from ultralytics import YOLO
import cv2
import numpy as np

output_path = 'out'
yolov8_model = 'yolov8n-seg.pt'

def main():
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        if not os.path.isdir(output_path):
            print("output folder is file, can't output results. Exit")
            sys.exit()

    # any video source opencv understands 
    # 0,1,... for usb cams, 'rtsp://...' for RTSP streams, /path/video.mp4 for video file
    video_source = sys.argv[1] 
    do_inferencing(video_source)
    
    #do_inferencing_single()



def do_inferencing(source):
    if source.isdigit():
        source = int(source)
    cap    = cv2.VideoCapture(source)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    start_time = datetime.datetime.now()
    time_prefix = start_time.strftime("%Y%m%d-%H%M")
    output_filename = create_ouput_filename(source)
    output_filename = 'out/' + time_prefix + "-" + yolov8_model + "-" + output_filename
    print(output_filename)
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))    

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", 800, 600)
    
    model = YOLO(yolov8_model)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            #res_plotted = segment_objects(frame, model) # if you want to work with prediction result
            prediction = model.predict(frame)
            res_plotted = prediction[0].plot()            
            cv2.imshow('output', res_plotted)

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

def segment_objects(frame, model):

    overlay = frame.copy() # overlay to add segmentation polygons
    alpha = 0.4  # Transparency factor.
    seg = model.predict(frame)
    names = seg[0].names    

    for idx, mask in enumerate(seg[0].masks.xy):
        conf = seg[0].boxes[idx].conf.data[0].item()
        if conf > 0.3:
            polyline = mask.astype(np.int32)
            predicted_class_id = seg[0].boxes[idx].cls.data[0].item()
            pred_class = names[predicted_class_id]
            x,y = centroid(polyline)
            COLOR = (0,int(255/(predicted_class_id+1)),int(255/(predicted_class_id+1)))
            COLOR2 = (50,int(255/(predicted_class_id+1)),int(255/(predicted_class_id+1)))
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

def do_inferencing_single():
    source = "sample_images/carmel01.png"
    model = YOLO(yolov8_model)
    img_raw = cv2.imread(source, cv2.IMREAD_COLOR)
    overlay = img_raw.copy() # overlay to add segmentation polygons
    alpha = 0.4  # Transparency factor.
    seg = model.predict(img_raw)
    names = seg[0].names    

    for idx, mask in enumerate(seg[0].masks.xy):
        conf = seg[0].boxes[idx].conf.data[0].item()
        if conf > 0.3:
            polyline = mask.astype(np.int32)
            predicted_class_id = seg[0].boxes[idx].cls.data[0].item()
            pred_class = names[predicted_class_id]
            x = np.mean(polyline[0,:])
            y = np.mean(polyline[1,:])
            print(str(x) + "," + str(y))
            #x,y = centroid(polyline)
            COLOR = (0,int(255/(predicted_class_id+1)),int(255/(predicted_class_id+1)))
            COLOR2 = (50,int(255/(predicted_class_id+1)),int(255/(predicted_class_id+1)))
            cv2.fillPoly(overlay,[polyline], COLOR)
            cv2.putText(
                overlay, 
                f'{pred_class} ({round(float(conf)*100,2)}%)', 
                (round(x), round(y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, 
                color=COLOR2, 
                thickness=5)

    img_out = cv2.addWeighted(overlay, alpha, img_raw, 1 - alpha, 0)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow('output', img_out)
    cv2.waitKey()
    cv2.destroyWindow("output") 

main()