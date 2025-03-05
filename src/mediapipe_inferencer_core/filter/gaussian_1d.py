import numpy as np
from scipy.ndimage import gaussian_filter1d
from .landmark_filter import ILandmarkFilter
from mediapipe_inferencer_core.data_class import Landmark, LandmarkList

class Gaussian1dFilter(ILandmarkFilter):
    def __init__(self, sigma:float, window_size:int, n_landmarks:int)->None:
        self._sigma = sigma
        self._window_size = window_size if window_size % 2 == 1 else window_size - 1
        self._n_landmarks = n_landmarks
        self._prev= []

    @property
    def result(self):
        pass

    def filter(self, current: LandmarkList)->LandmarkList:
        if current is None:
            return current
        self._push(current)
        # filtering with Gaussian 1d
        center = len(self._prev) // 2
        view = np.array(self._prev).transpose(2, 1, 0)
        filtered_x          = [gaussian_filter1d(x, self._sigma)[center] for x in view[0, :, :]]
        filtered_y          = [gaussian_filter1d(y, self._sigma)[center] for y in view[1, :, : ]]
        filtered_z          = [gaussian_filter1d(z, self._sigma)[center] for z in view[2, :, :]]
        filtered_confidence = [gaussian_filter1d(confidence, self._sigma)[center] for confidence in view[3, :, :]]
        # packing results as LandmarkList
        filtered_result = LandmarkList(
                [
                    Landmark(filtered_x[i], filtered_y[i], filtered_z[i], filtered_confidence[i]) for i in range(self._n_landmarks)
                ]
            )
        return filtered_result

    def _push(self, result: LandmarkList)->None:
        current = np.copy(result.values)
        self._prev.append(current)
        if len(self._prev) > self._window_size:
            self._pop()

    def _pop(self)->LandmarkList:
        self._prev = self._prev[1:]