from ultralytics.yolo.engine.results import Results
from collections import defaultdict, Counter
import numpy as np

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
    
