import csv
import json
from collections import Counter, defaultdict

import numpy as np
from ultralytics.yolo.engine.results import Results


class SpeedStats:

    _SPEED_KEYS = ('preprocess', 'inference', 'postprocess')

    def __init__(self) -> None:
        self.stats = defaultdict(list)

    def append(self, results: Results):
        for key in self._SPEED_KEYS:
            self.stats[key].append(results.speed[key])

    def get_mov_avg(self, window_size):
        avgs = dict()
        for key in self._SPEED_KEYS:
            avgs[key] = self._mov_avg(self.stats[key], window_size)
        return avgs
    
    @classmethod
    def _mov_avg(cls, values, window_size):
        return np.average(values[-window_size:])
    
class DetectionStatsWriter:

    def __init__(self, output_file, classes) -> None:
        with open('yolov8_classes.json', 'r') as f:
            self.yolo_id_to_class = {int(id_): class_ for id_, class_ in json.load(f).items()}
        self.yolo_class_to_id = {class_: id_ for id_, class_ in self.yolo_id_to_class.items()}
        self.classes = classes
        self.output_file = open(output_file, 'w')

        fieldnames = ['frame_no', 'frame_timestamp', *classes]
        self.csv_writer = csv.writer(self.output_file)

        # Write header
        self.csv_writer.writerow(fieldnames)

    def write(self, frame_no, frame_timestamp, results: Results):
        self.csv_writer.writerow([
            frame_no,
            frame_timestamp,
            *self._extract_class_counts(results)
        ])

    def _extract_class_counts(self, results: Results):
        count_dict = Counter(results.boxes.cls.int().tolist())
        return [count_dict[self.yolo_class_to_id[class_]] for class_ in self.classes]

    def close(self):
        self.output_file.close()