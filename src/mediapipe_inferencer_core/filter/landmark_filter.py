from abc import ABC, abstractclassmethod, abstractproperty
from mediapipe_inferencer_core.data_class import LandmarkList

class ILandmarkFilter(ABC):
    @abstractclassmethod
    def filter(self, results:LandmarkList):
        pass
    @abstractproperty
    def result(self):
        pass