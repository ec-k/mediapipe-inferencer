from mediapipe_inferencer_core.data_class.landmark import Landmark
from mediapipe_inferencer_core.filter.landmark_filter import LandmarkFilter
from mediapipe_inferencer_core.filter.exponential_smoothing import ExponentialSmoothing
import math

class OneEuroFilter(LandmarkFilter):
    def __init__(self, min_cutoff:float, slope:float, d_cutoff:float) -> None:
        super().__init__(filter_length=1)
        self.__prev_time = 0
        self.__min_cutoff = min_cutoff
        self.__slope = slope
        self.__d_cutoff = d_cutoff
        self.__position_filter = PerLandmarkFilter()
        self.__velocity_filter = PerLandmarkFilter()
        self.__is_first = True
        self.__is_first_2 = True

    def filter(self, current:list[Landmark], time:float) -> list[Landmark]:
        if current is None:
            return
        if self.__is_first_2:
            self.__position_filter.init_cache(current)
            self.__velocity_filter.init_cache([Landmark(0,0,0,1) for i in range(len(current))])
            self.__is_first_2 = False
        if current is None:
            return current
        t_e = time - self.__prev_time
        if t_e < 1e-5:
            return current
        update_rate = 1/t_e
        filtered = [self._filter(current[i], i, update_rate) for i in range(len(current))]
        self._cached_result = filtered
        self.__prev_time = time
        return self._cached_result

    def _filter(self, current:Landmark, index:int, update_rate:float) -> Landmark:
        dx = Landmark(0,0,0,1)
        if self.__is_first:
            self.__is_first = False
        else:
            dx = Landmark.multiply(Landmark.sub(current, self.__position_filter.result[index]), update_rate)
        alpha = OneEuroFilter.__alpha(update_rate, self.__d_cutoff)
        edx = ExponentialSmoothing._filter_per_landmark(dx, self.__position_filter.result[index], alpha)
        self.__velocity_filter.update(edx, index)
        cutoff = self.__min_cutoff + self.__slope * Landmark.magnitude(edx.position)
        alpha = OneEuroFilter.__alpha(update_rate, cutoff)
        filtered = ExponentialSmoothing._filter_per_landmark(current,self.__position_filter.result[index], alpha)
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
        self._cached_result[index] = result
        self._prev_results[0][index] = result

    def init_cache(self, new_result:list[Landmark])->None:
        self._cached_result = [Landmark() for i in range(len(new_result))]
        self._push(new_result)