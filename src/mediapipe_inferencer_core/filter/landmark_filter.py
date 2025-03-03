from abc import ABC, abstractclassmethod, abstractproperty
from mediapipe_inferencer_core.data_class.landmark import Landmark

class ILandmarkFilter(ABC):
    @abstractclassmethod
    def filter(self, results:list[Landmark]):
        pass
    @abstractproperty
    def result(self):
        pass