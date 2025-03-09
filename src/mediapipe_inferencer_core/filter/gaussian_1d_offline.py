import numpy as np
from scipy.ndimage import gaussian_filter1d
from .landmark_filter import ILandmarkFilter
from mediapipe_inferencer_core.data_class import Landmark, LandmarkList

class Gaussian1dFilterOffline():
    def __init__(self, sigma:float, window_size:int, n_landmarks:int)->None:
        self._sigma = sigma
        self._window_size = window_size if window_size % 2 == 1 else window_size - 1
        self._n_landmarks = n_landmarks

    @property
    def result(self):
        pass

    def filter(self, results: np.ndarray)->np.ndarray:
        if results is None:
            return results

        # 1. clone result
        # 2. apply gaussian convolution to cloned result
            # Just you apply gaussian_filter1d, then you can get convuted result
        # 3. return it
        
        # current_i
        # get result: [max(current_i - window//2, 0), min(current_i + window//2, num_results)]
        # filtered[i] = [gaussian_filter1d(value, self._sigma)]
        
        # filtering with Gaussian 1d
        filtered = np.ndarray(shape=results.shape)
        filtered_view = filtered.transpose(1,0)
        n_value = len(results[0])
        view = np.array(results).transpose(1, 0)
        for value_i in range(n_value):
            filtered_view[value_i] = gaussian_filter1d(view[value_i], self._sigma, radius=self._window_size//2)
        return filtered