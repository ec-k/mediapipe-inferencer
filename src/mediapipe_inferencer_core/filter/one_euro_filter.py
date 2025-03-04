from mediapipe_inferencer_core.data_class import Landmark, LandmarkList
from mediapipe_inferencer_core.filter.landmark_filter import ILandmarkFilter
from mediapipe_inferencer_core.filter.exponential_smoothing import ExponentialSmoothing
import math
import numpy as np

class OneEuroFilter(ILandmarkFilter):
    def __init__(self, min_cutoff:float, slope:float, d_cutoff:float) -> None:
        super().__init__()
        self.__prev_time = 0
        self.__min_cutoff = min_cutoff
        self.__slope = slope
        self.__d_cutoff = d_cutoff
        self.__position_filter = PerLandmarkFilter()
        self.__velocity_filter = PerLandmarkFilter()
        self.__is_first = True
        self.__is_first_2 = True

    @property
    def result(self)-> LandmarkList:
        return self.__position_filter.result

    def filter(self, current:LandmarkList, time:float) -> LandmarkList:
        if current is None:
            return current
        if self.__is_first_2:
            self.__position_filter.init_cache(current)
            self.__velocity_filter.init_cache(LandmarkList([Landmark(0,0,0,1) for _ in range(len(current.values))]))
            self.__is_first_2 = False
        t_e = time - self.__prev_time
        if t_e < 1e-5:
            return current
        update_rate = 1/t_e
        filtered = LandmarkList([self._filter(current.value(i), i, update_rate) for i in range(len(current.values))])
        self._cached_result = filtered
        self.__prev_time = time
        return self._cached_result

    def _filter(self, current:Landmark, index:int, update_rate:float) -> Landmark:
        dx = Landmark(0,0,0,1)
        if self.__is_first:
            self.__is_first = False
        else:
            dx = Landmark.multiply(Landmark.sub(current, self.__position_filter.result.value(index)), update_rate)
        alpha = OneEuroFilter.__alpha(update_rate, self.__d_cutoff)
        edx = ExponentialSmoothing._filter_per_landmark(dx, self.__position_filter.result.value(index), alpha)
        self.__velocity_filter.update(edx, index)
        cutoff = self.__min_cutoff + self.__slope * Landmark.magnitude(edx.position)
        alpha = OneEuroFilter.__alpha(update_rate, cutoff)
        filtered = ExponentialSmoothing._filter_per_landmark(current,self.__position_filter.result.value(index), alpha)
        self.__position_filter.update(filtered, index)
        return filtered

    def __alpha(update_rate:float, min_cutoff:float) -> float:
        time_constant = 1 / (2*math.pi*min_cutoff)
        return 1 / (1 + time_constant*update_rate)


class PerLandmarkFilter(ExponentialSmoothing):
    def __init__(self):
        dummy = 1
        super().__init__(dummy)

    def update(self, result:Landmark, index:int)->None:
        tmp = np.array([result.x, result.y, result.z, result.confidence])
        self._prev_results.values[index] = tmp

    def init_cache(self, new_result:LandmarkList)->None:
        self._update_result_cache(new_result)