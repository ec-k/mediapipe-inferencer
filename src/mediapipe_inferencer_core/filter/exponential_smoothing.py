from mediapipe_inferencer_core.data_class.landmark import Landmark
from mediapipe_inferencer_core.filter.landmark_filter import LandmarkFilter

class ExponentialSmoothing(LandmarkFilter):
    def __init__(self, smoothing_factor):
        super().__init__(filter_length=1)
        self.__smoothing_factor = smoothing_factor

    def filter(self, results:list[Landmark]):
        if results is None:
            return
        filtered =  self._filter(results, self._prev_results)
        self._cached_result = filtered
        self._push(filtered)
        if len(self._prev_results) > self._filter_length:
            self._pop()
        return self._cached_result

    def _filter(self, current, prev):
        if prev is None or len(prev) < 1:
            return current
        result = [Landmark.lerp(prev[0][i], current[i], self.__smoothing_factor) for i in range(len(current))]
        return result