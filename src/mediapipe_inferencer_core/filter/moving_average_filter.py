from collections import deque
from mediapipe.framework.formats import landmark_pb2

class MovingAverage:
    def __init__(self, filter_length, result_update_method):
        self._prev_results = deque()
        self._filter_length = filter_length
        self._cached_result = None
        self._result_update_method = result_update_method

    @property
    def result(self):
        return self._cached_result

    def update(self, results):
        self._push(results)
        if len(self._prev_results) <= self._filter_length:
            return
        self._pop()
        self._cached_result = self._result_update_method(self._prev_results)

    def _push(self, results):
        self._prev_results.append(results)

    def _pop(self):
        return self._prev_results.popleft()

def no_process(prev_results):
    return prev_results[0]