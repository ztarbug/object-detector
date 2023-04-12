import sys
import os
import datetime

from pprint import pprint
from ultralytics import YOLO
import cv2

output_path = 'out'
yolov8_model = 'yolov8s.pt'

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
            #res_plotted = detect_objects(frame, model) # if you want to work with prediction result
            prediction = model.predict(frame)
            res_plotted = prediction[0].plot()
            cv2.imshow('output', res_plotted)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break 
            out.write(res_plotted)
        else:
            print("couldn't read next frame")
            break
    
    end_time = datetime.datetime.now()
    print("Finished post processing in " + str(end_time - start_time))
    cap.release()
    cv2.destroyAllWindows()

def detect_objects(img, model):
    BBOX_COLOR = (255, 0, 0)

    #det = model.predict(img,device="cpu")
    det = model.predict(img)
    boxes = det[0].boxes
    names = det[0].names

    i = 0
    for box in boxes:
        predicted_class_id = box.cls.data[0].item()
        pred_class = names[predicted_class_id]
        conf = box.conf.data[0].item()
        xyboxes = box.xyxy.data[0]
        x1 = int(xyboxes.data[0].item())
        y1 = int(xyboxes.data[1].item())
        x2 = int(xyboxes.data[2].item())
        y2 = int(xyboxes.data[3].item())

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
        cv2.putText(
            img, 
            f'{pred_class} ({round(float(conf)*100,2)}%)', 
            (x1+3, y1-5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1, 
            color=BBOX_COLOR, 
            thickness=5)
    return img

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

main()