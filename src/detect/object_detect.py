import argparse
import os
import re
import time
from os.path import basename, isfile, join, splitext

import cv2
import numpy as np
import tensorflow as tf
from modelfiles import get_model_dir

BBOX_COLOR = (255, 0, 0)
IMAGE_FILE_EXTENSIONS = ['.jpg', '.png']


def load_image(path):
  image_bgr = cv2.imread(path)
  b, g, r = cv2.split(image_bgr)
  image_rgb = cv2.merge([r, g, b])
  return image_rgb

def predict(model, image, min_score):
  prediction = model(np.expand_dims(image, 0))
  filter_mask = prediction['detection_scores'][0] > min_score
  return {
    'scores': prediction['detection_scores'][0][filter_mask],
    'classes': prediction['detection_classes'][0][filter_mask],
    'boxes': prediction['detection_boxes'][0][filter_mask]
  }

def tf_box_to_bbox(image, prediction_box):
  shape = tf.shape(image)
  return (
    tf_coord_to_pixel(prediction_box[0], shape[0]),
    tf_coord_to_pixel(prediction_box[1], shape[1]),
    tf_coord_to_pixel(prediction_box[2], shape[0]),
    tf_coord_to_pixel(prediction_box[3], shape[1])
  )

# prediction_boxes are made of float32 tensors indicating a ratio from 0 to 1 so we have to calculate the pixel position
def tf_coord_to_pixel(coord, dim_size):
  return int(tf.cast(coord * tf.cast(dim_size, tf.float32), tf.int32))

def draw_bbox(image, bbox, class_name, score):
  ymin, xmin, ymax, xmax = bbox

  avg_shape = int(tf.cast(min(tf.shape(image).numpy()[:2]), tf.int32))
  scaling_factor = max(1, int(round(avg_shape * 0.001, 0)))
  
  cv2.rectangle(image, (xmin, ymin), (xmax, ymax), BBOX_COLOR, thickness=2*scaling_factor)
  cv2.putText(
    image, 
    f'{class_name} ({round(float(score)*100,2)}%)', 
    (xmin+3*scaling_factor, ymax-5*scaling_factor), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    fontScale=1*scaling_factor, 
    color=BBOX_COLOR, 
    thickness=2*scaling_factor
  )

def save_image(image_rgb, path):
  r, g, b = cv2.split(image_rgb)
  image_bgr = cv2.merge([b, g, r])
  cv2.imwrite(path, image_bgr)

def get_images_in_path(path):
  image_files = filter(lambda p: isfile(join(path, p)) and splitext(p)[1] in IMAGE_FILE_EXTENSIONS, os.listdir(path))
  return list(map(lambda f: join(path, f), image_files))

COCO_CLASSNAMES = {
  0: u'__background__',
  1: u'person',
  2: u'bicycle',
  3: u'car',
  4: u'motorcycle',
  5: u'airplane',
  6: u'bus',
  7: u'train',
  8: u'truck',
  9: u'boat',
  10: u'traffic light',
  11: u'fire hydrant',
  12: u'stop sign',
  13: u'parking meter',
  14: u'bench',
  15: u'bird',
  16: u'cat',
  17: u'dog',
  18: u'horse',
  19: u'sheep',
  20: u'cow',
  21: u'elephant',
  22: u'bear',
  23: u'zebra',
  24: u'giraffe',
  25: u'backpack',
  26: u'umbrella',
  27: u'handbag',
  28: u'tie',
  29: u'suitcase',
  30: u'frisbee',
  31: u'skis',
  32: u'snowboard',
  33: u'sports ball',
  34: u'kite',
  35: u'baseball bat',
  36: u'baseball glove',
  37: u'skateboard',
  38: u'surfboard',
  39: u'tennis racket',
  40: u'bottle',
  41: u'wine glass',
  42: u'cup',
  43: u'fork',
  44: u'knife',
  45: u'spoon',
  46: u'bowl',
  47: u'banana',
  48: u'apple',
  49: u'sandwich',
  50: u'orange',
  51: u'broccoli',
  52: u'carrot',
  53: u'hot dog',
  54: u'pizza',
  55: u'donut',
  56: u'cake',
  57: u'chair',
  58: u'couch',
  59: u'potted plant',
  60: u'bed',
  61: u'dining table',
  62: u'toilet',
  63: u'tv',
  64: u'laptop',
  65: u'mouse',
  66: u'remote',
  67: u'keyboard',
  68: u'cell phone',
  69: u'microwave',
  70: u'oven',
  71: u'toaster',
  72: u'sink',
  73: u'refrigerator',
  74: u'book',
  75: u'clock',
  76: u'vase',
  77: u'scissors',
  78: u'teddy bear',
  79: u'hair drier',
  80: u'toothbrush'
}


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--image-dir', help='Directory where to look for images to annotate (default: "./images")', default='./images')
  parser.add_argument('-m', '--model-dir', help='Path to the "saved_model" directory of the model to run (default: a directory starting with "efficientdet_d")')
  parser.add_argument('-o', '--output-dir', help='Directory where to put annotated images (default: "./annotated_images")', default='./annotated_images')
  parser.add_argument('-c', '--min-confidence', help='Only include objects if the confidence of detection is higher than this (range 0-100 percent)', default=30, type=int)
  parser.add_argument('-s', '--model-size', help='Which EfficientDet model size should be downloaded (range [0..7])', default=0, type=int)
  parser.add_argument('-d', '--download', help='Should the EfficientDet model be automatically downloaded if not found in the provided directory?', default=False, type=bool)
  args = vars(parser.parse_args())

  if args['model_dir'] is None:
    SAVED_MODEL_DIR = os.path.join(get_model_dir(model_size=args['model_size'], auto_download=args['download']), 'saved_model')
  else:
    SAVED_MODEL_DIR = os.path.join(get_model_dir(model_dir=args['model_dir'], model_size=args['model_size'], auto_download=args['download']), 'saved_model')

  OUTPUT_DIR = args['output_dir']
  IMAGES = get_images_in_path(args['image_dir'])
  MIN_SCORE = args['min_confidence']/100

  os.makedirs(OUTPUT_DIR, exist_ok=True)

  eff_det = tf.saved_model.load(SAVED_MODEL_DIR)


  for image_path in IMAGES:
    image = load_image(image_path)
    start_time = time.time()
    prediction = predict(eff_det, image, MIN_SCORE)
    print(f'Inference time: {time.time() - start_time} s')

    for prediction_box, class_id, score in zip(prediction['boxes'], prediction['classes'], prediction['scores']):
      bbox = tf_box_to_bbox(image, prediction_box)
      draw_bbox(image, bbox, COCO_CLASSNAMES[int(class_id)], score)

    output_path = join(OUTPUT_DIR, basename(image_path))
    save_image(image, output_path)
    print(f'Saved annotated image to {output_path}')

  