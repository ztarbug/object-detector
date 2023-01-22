# How to use EfficientDet

## Prerequisites
- Install opencv on your system for the Python bindings to work\
  `sudo apt install libopencv-dev`
- Create and activate virtual Python environment if you like (e.g. `python3 -m venv .venv`, followed by `source .venv/bin/activate`)
- Install dependencies\
  `pip install -r requirements.txt`

## Download and prepare model
- Download EfficientDet model of desired size from the TF2 detection zoo\
  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

- Extract .tar.gz file
  `tar xvf efficientdet_d*.tar.gz`

## Annotate some images
- Put some images into `./images` or somewhere else (if you choose a custom directory you have to supply it to the script below)
- Run `object_detect.py` (Run it with `--help` to get some usage hints)
- Find your annotated images in `./annotated_images` (by default)