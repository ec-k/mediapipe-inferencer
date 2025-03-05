import numpy as np
from .landmark import Landmark

class LandmarkList:
    def __init__(self, listed_data: list[Landmark]):
        self._value = np.array([[data.x, data.y, data.z, data.confidence] for data in listed_data])

    def value(self, index:int)->Landmark:
        return Landmark(self._value[index, 0], self._value[index, 1],self._value[index, 2],self._value[index, 3])

    @property
    def values(self) -> np.ndarray:
        return self._value