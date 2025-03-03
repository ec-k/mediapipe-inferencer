from mediapipe_inferencer_core.data_class.landmark import Landmark
from mediapipe_inferencer_core.filter.landmark_filter import LandmarkFilter

class ExponentialSmoothing(LandmarkFilter):
    def __init__(self, smoothing_factor):
        super().__init__(filter_length=1)
        self.__smoothing_factor = smoothing_factor

    def filter(self, results:list[Landmark]):
        if results is None:
            return
        filtered = results if len(self._prev_results) < 1 else self._filter(results, self._prev_results[0], self.__smoothing_factor)
        self._cached_result = filtered
        self._push(filtered)
        if len(self._prev_results) > self._filter_length:
            self._pop()
        return self._cached_result

    def _filter(self, current:list[Landmark], prev:list[Landmark], smoothing_factor:float)->list[Landmark]:
        if prev is None or len(prev) < 1:
            return current
        result = [ExponentialSmoothing._filter_per_landmark(current[i], prev[i], smoothing_factor) for i in range(len(current))]
        return result

    def _filter_per_landmark(current: Landmark, prev: Landmark, smoothing_factor:float)->Landmark:
        return Landmark.lerp(prev, current, smoothing_factor)