from mediapipe_inferencer_core.data_class import LandmarkList, Landmark
from mediapipe_inferencer_core.filter.landmark_filter import ILandmarkFilter

class ExponentialSmoothing(ILandmarkFilter):
    def __init__(self, smoothing_factor):
        super().__init__()
        self.__smoothing_factor = smoothing_factor
        self._prev_results = None

    @property
    def result(self)->LandmarkList:
        return self._prev_results

    def filter(self, results:LandmarkList):
        if results is None:
            return
        filtered = results if self._prev_results is None else self._filter(results, self._prev_results, self.__smoothing_factor)
        self._update_result_cache(filtered)
        return filtered

    def _filter(self, current:LandmarkList, prev:LandmarkList, smoothing_factor:float)->LandmarkList:
        if prev is None or len(prev.values) < 1:
            return current
        result = LandmarkList([ExponentialSmoothing._filter_per_landmark(current.value(i), prev.value(i), smoothing_factor) for i in range(len(current.values))])
        return result

    def _filter_per_landmark(current: Landmark, prev: Landmark, smoothing_factor:float)->Landmark:
        return Landmark.lerp(prev, current, smoothing_factor)

    def _update_result_cache(self, result: LandmarkList):
        self._prev_results = result