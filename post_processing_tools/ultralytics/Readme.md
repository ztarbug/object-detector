# Yolov8 / Ultralytics post processing
Here are some post processing tools, that make use of the awesome Ultralytics library. Please refer for more details to their repo:
https://github.com/ultralytics/ultralytics

## Prerequisites
- OpenCV installed on OS
- Poetry package manager installed \
  `curl -sSL https://install.python-poetry.org | python3 -` (see https://python-poetry.org/docs/#installing-with-the-official-installer)

## How to run
This lib makes use of PyTorch, please refer to PyTorch, which hardware is supported. Code here was successfully tested running on CPU and CUDA/GPU. 

    poetry install
    poetry run ./ultra-process.py /dev/video0
    # or enter a venv shell
    poetry shell
    ./ultra-process.py /dev/video0

Run `./ultra-process.py -h` for usage hints

You can provide an integers to select USB cam, a RTSP url or a path to a video file. Script will then run and display inferencing results and also writes a file to folder out.

