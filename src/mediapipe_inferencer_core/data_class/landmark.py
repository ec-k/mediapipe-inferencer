import numpy as np
from mediapipe_inferencer_core.util import float_util

class Landmark:
    def __init__(self, x=0, y=0, z=0, confidence=0):
        self.__value = np.array([x, y, z, confidence])
    @property
    def position(self):
        return self.__value[0:3]
    @property
    def x(self):
        return self.__value[0]
    @property
    def y(self):
        return self.__value[1]
    @property
    def z(self):
        return self.__value[2]
    @property
    def confidence(self):
        return self.__value[3]
    @property
    def Landmark(self):
        return self.__value

    def sub(lm_1:Landmark, lm_2:Landmark)->Landmark:
        pos = np.subtract(lm_1.position, lm_2.position)
        confidence = min(lm_1.confidence, lm_2.confidence)
        return Landmark(pos[0], pos[1], pos[2], confidence)

    def multiply(lm:Landmark, scalar:float)->Landmark:
        pos = np.multiply(lm.position, scalar)
        return Landmark(pos[0], pos[1], pos[2], lm.confidence)
    
    def magnitude(lm:Landmark)->float:
        return np.linalg.norm(lm[0:3])

    def lerp(from_lm: Landmark, to_lm: Landmark, lerp_amount: float) -> Landmark:
        amount = float_util.clamp(lerp_amount, 0, 1)
        np_result = np.multiply(to_lm.Landmark, amount) + np.multiply(from_lm.Landmark, 1-amount)
        return Landmark(np_result[0], np_result[1], np_result[2], np_result[3])

    def value_equal(lm_1: Landmark, lm_2: Landmark) -> bool:
        return lm_1.x == lm_2.x \
            and lm_1.y == lm_2.y \
            and lm_1.z == lm_2.z \
            and lm_1.confidence == lm_2.confidence
