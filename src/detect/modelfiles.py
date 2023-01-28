import os
import logging
import shutil
import requests
import re
import tempfile

MODEL_PATH_PATTERN = re.compile(r'^efficientdet_d.*')

def build_efficientdet_url(size):
    assert size >= 0 or size <= 7, f'model size {size} out of bounds [0..7]'
    return f'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d{size}_coco17_tpu-32.tar.gz'

def efficientdet_dir_name(size):
    return f'efficientdet_d{size}_coco17_tpu-32'

def find_model_dir(ref_path='.'):
    path = next(filter(lambda p: os.path.isdir(p) and MODEL_PATH_PATTERN.match(p), os.listdir(ref_path)), None)
    return path

def download_file(url, dir='.'):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(os.path.join(dir, local_filename), 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return local_filename

def get_model_dir(auto_download = False, model_dir='.', model_size=0):
    dir = find_model_dir(model_dir)
    if dir is None and auto_download:
        logging.info(f'No model found in dir {model_dir}. Downloading...')
        tempdir = tempfile.gettempdir()
        archive_file = download_file(build_efficientdet_url(model_size))
        shutil.unpack_archive(archive_file, extract_dir=tempdir)
        return os.path.join(tempdir, efficientdet_dir_name(model_size))
    elif dir is None: 
        logging.error(f'Expected model directory matching "{MODEL_PATH_PATTERN.pattern}" in {model_dir}')
        return None
    else:
        return dir

    