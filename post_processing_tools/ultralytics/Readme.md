# Yolov8 / Ultralytics post processing
Here are some post processing tools, that make use of the awesome Ultralytics library. Please refer for more details to their repo:
https://github.com/ultralytics/ultralytics

## How to run
This lib makes use of PyTorch, please refer to PyTorch, which hardware is supported. Code here was successfully tested running on CPU and CUDA/GPU. 

    pip install -r requirements
    python3 ultra-***.py source

You can provide an integers to select USB cam, a RTSP url or a path to a video file. Script will then run and display inferencing results and also writes a file to folder out.
### Examples

    # process a video file
    python ultra-detection.py path/to/video.mp4  

    # use first USB cam
    python ultra-detection.py 0  

    # process rtsp stream
    python ultra-detection.py rtsp://localhost:554/stream

