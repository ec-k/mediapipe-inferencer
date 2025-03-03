from mediapipe_inferencer_core.data_class.landmark import Landmark
from mediapipe_inferencer_core.filter.landmark_filter import ILandmarkFilter

class ExponentialSmoothing(ILandmarkFilter):
    def __init__(self, smoothing_factor):
        super().__init__()
        self.__smoothing_factor = smoothing_factor
        self._prev_results = None

    @property
    def result(self)->list[Landmark]:
        return self._prev_results

    def filter(self, results:list[Landmark]):
        if results is None:
            return
        filtered = results if self._prev_results is None else self._filter(results, self._prev_results, self.__smoothing_factor)
        self._update_result_cache(filtered)
        return filtered

    def _filter(self, current:list[Landmark], prev:list[Landmark], smoothing_factor:float)->list[Landmark]:
        if prev is None or len(prev) < 1:
            return current
        result = [ExponentialSmoothing._filter_per_landmark(current[i], prev[i], smoothing_factor) for i in range(len(current))]
        return result

    def _filter_per_landmark(current: Landmark, prev: Landmark, smoothing_factor:float)->Landmark:
        return Landmark.lerp(prev, current, smoothing_factor)

    def _update_result_cache(self, result: list[Landmark]):
        self._prev_results = result