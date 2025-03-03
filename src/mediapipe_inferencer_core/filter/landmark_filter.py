from abc import ABC, abstractclassmethod
from collections import deque
from mediapipe_inferencer_core.data_class.landmark import Landmark

class LandmarkFilter(ABC):
    def __init__(self, filter_length):
        self._prev_results = deque()
        self._filter_length = filter_length
        self._cached_result = None

    @abstractclassmethod
    def filter(self, results:list[Landmark]):
        pass

    @property
    def result(self):
        return self._cached_result

    def _push(self, results):
        self._prev_results.append(results)

    def _pop(self):
        return self._prev_results.popleft()